import numpy as np
import scipy as sp
import matplotlib as mp

# Try to use a GUI backend (Qt). If not available, fall back silently.
try:
    mp.use("QtAgg")
except Exception:
    pass

from matplotlib import pyplot as plt
import skimage as ski
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal.windows as wind
from matplotlib.widgets import SpanSelector, Button
import os

# ------------------------------------------------
# 1. Load image and reconstruct 1D signal (sig_dig)
# ------------------------------------------------

# Ask user for file name (path is fixed here like in your notebook)
path = "C:/Users/User/Pictures/Aspa"
file = input("Введите имя файла с указанием формата: ")
filename = os.path.join(path, file)

# Read image (oscillogram)
signal = ski.io.imread(filename)

# Ask user for RGB color of the line
pixcolor = []
rgb = ["r", "g", "b"]
print("Введите цвет линии в формате rgb \n")
for j in range(3):
    pixcolor.append(int(input(rgb[j] + " : ")))

# Build mask where pixels match the given color
masks = np.zeros((signal.shape[0], signal.shape[1]), dtype=bool)
for j in range(signal.shape[1]):
    for i in range(signal.shape[0]):
        masks[i, j] = (
            signal[i, j, 0] == pixcolor[0]
            and signal[i, j, 1] == pixcolor[1]
            and signal[i, j, 2] == pixcolor[2]
        )

# Reconstruct 1D digital signal sig_dig from image
rows = np.arange(signal.shape[0])
sig_dig = np.zeros(signal.shape[1])

for j in range(signal.shape[1]):
    mask_col = masks[:, j]
    if np.any(mask_col):
        # Same formula as in your notebook
        sig_dig[j] = (
            signal.shape[0]
            - np.ceil(np.mean(rows[mask_col]))
            - np.floor(signal.shape[0] / 2)
        ) / signal.shape[0] * 2
    else:
        sig_dig[j] = 0.0

sig_dig[np.isnan(sig_dig)] = 0

# ------------------------------------------------
# 2. Interactive editor: zero out intervals with undo/reset/confirm
# ------------------------------------------------

# Keep a copy for reset
original_sig_dig = sig_dig.copy()

# Undo stack
undo_stack = []

# Create interactive figure
fig, ax = plt.subplots(figsize=(12, 4))
plt.subplots_adjust(bottom=0.25)

line, = ax.plot(sig_dig, color='blue')
ax.set_title("Drag to select an interval to zero out")


def onselect(xmin, xmax):
    """SpanSelector callback: zeroes selected interval."""
    global sig_dig, undo_stack

    i_min = int(max(0, np.floor(xmin)))
    i_max = int(min(len(sig_dig), np.ceil(xmax)))

    if i_min >= i_max:
        return  # ignore too small/no selections

    # Save state for undo
    undo_stack.append(sig_dig.copy())

    print(f"Selected interval: {i_min} → {i_max}")

    # Zero out selected region
    sig_dig[i_min:i_max] = 0

    # Update plot
    line.set_ydata(sig_dig)
    fig.canvas.draw_idle()


def undo(event):
    """Undo last modification."""
    global sig_dig, undo_stack

    if not undo_stack:
        print("Undo stack empty.")
        return

    sig_dig = undo_stack.pop()
    line.set_ydata(sig_dig)
    fig.canvas.draw_idle()
    print("Undo performed.")


def reset(event):
    """Reset signal to original."""
    global sig_dig, undo_stack, original_sig_dig

    sig_dig = original_sig_dig.copy()
    undo_stack.clear()

    line.set_ydata(sig_dig)
    fig.canvas.draw_idle()
    print("Signal reset to original.")


def confirm(event):
    """Confirm edits and close window."""
    print("Confirmed. Window will close. Edited signal is preserved in variable 'sig_dig'.")
    plt.close(fig)


# Buttons
undo_ax = fig.add_axes([0.10, 0.05, 0.1, 0.075])
reset_ax = fig.add_axes([0.25, 0.05, 0.1, 0.075])
confirm_ax = fig.add_axes([0.75, 0.05, 0.15, 0.075])

undo_button = Button(undo_ax, "Undo")
undo_button.on_clicked(undo)

reset_button = Button(reset_ax, "Reset")
reset_button.on_clicked(reset)

confirm_button = Button(confirm_ax, "Confirm")
confirm_button.on_clicked(confirm)

# Span selector
span = SpanSelector(
    ax,
    onselect,
    direction='horizontal',
    useblit=True,
    props=dict(alpha=0.3, facecolor='red'),
    interactive=True,
)

# This show is blocking: after you press "Confirm" and the window closes,
# the script continues to the next plots.
plt.show()

# ------------------------------------------------
# 3. Plot final edited signal
# ------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(sig_dig, color='b', linewidth=2)
plt.xlabel("Sample index")
plt.ylabel("Amplitude")
plt.title("Отредактированный сигнал")
plt.grid(True)
plt.show()

# ------------------------------------------------
# 4. Spectrum computation and plots (as in the notebook)
# ------------------------------------------------

N = len(sig_dig)
matrix = np.zeros((5, N), dtype='float64')
t_matrix = np.zeros((5, N), dtype='float64')

# Time axis (based on your notebook's logic)
# You had t_beginning = 13.2e-6, t_end = 24.0e-6
t_beginning = 13.2e-6
t_end = 24.0e-6
t_max = t_end - t_beginning
t = np.linspace(-t_max / 2, t_max / 2, N)
delta_t = abs(t[1] - t[0])

# Spectrum
spectrum = fftshift(fft(fftshift(sig_dig))) / N
spect_f = fftfreq(N, delta_t) * 2 * np.pi

matrix[0, :] = sig_dig
t_matrix[0, :] = t

matrix[1, :] = np.abs(spectrum)
t_matrix[1, :] = fftshift(spect_f)
# matrix[2, :] = np.angle(spectrum)  # alternative

# In your notebook you effectively forced real/imag parts to constants
matrix[3, :] = np.real(spectrum)
matrix[3, :] = 1e-5
t_matrix[3, :] = fftshift(spect_f)

matrix[4, :] = np.imag(spectrum)
t_matrix[4, :] = fftshift(spect_f)
matrix[4, :] = 0

# Phase from real/imag (here always zero because of constants, like in notebook)
matrix[2, :] = np.arctan2(matrix[4, :], matrix[3, :])
t_matrix[2, :] = fftshift(spect_f)

# ------------------------------------------------
# 5. Final spectrum figure (3 subplots)
# ------------------------------------------------

fig2, axs = plt.subplots(3, 1, figsize=(17, 24))
fig2.suptitle("Спектры сигналов 90 градусов", y=0.93, fontsize=30)
plt.subplots_adjust(hspace=0.4)

Na = N // 2
mid = N // 2

# Original signal vs time
axs[0].plot(t_matrix[0, :], matrix[0, :], color='b', linewidth=2)
axs[0].set_title("Исходный сигнал", fontsize=20)
axs[0].set_xlabel("t, мкс", loc='center')
axs[0].grid(True)

# Amplitude spectrum
axs[1].plot(t_matrix[1, mid:mid + Na], matrix[1, mid:mid + Na], color='b', linewidth=2)
axs[1].set_title("Амплитудный спектр", fontsize=20)
axs[1].set_xlabel("f, МГц", loc='center')
axs[1].grid(True)

# Phase spectrum
axs[2].plot(t_matrix[2, mid:mid + Na], matrix[2, mid:mid + Na], color='b', linewidth=2)
axs[2].set_title("Фазовый спектр", fontsize=20)
axs[2].set_xlabel("f, МГц", loc='center')
axs[2].grid(True)

plt.show()
