#!/usr/bin/env python3
import sys
import board
import neopixel

# LED strip configuration
LED_COUNT = 15        # Change if you have more
LED_PIN = board.D18   # GPIO18 PWM pin
BRIGHTNESS = 0.5

pixels = neopixel.NeoPixel(
    LED_PIN,
    LED_COUNT,
    brightness=BRIGHTNESS,
    auto_write=True,
    pixel_order=neopixel.GRB
)

def set_color(color_tuple):
    pixels.fill(color_tuple)

# -----------------------
# Command Line Parsing
# -----------------------
if len(sys.argv) < 2:
    print("Usage:")
    print("  python3 leds.py red")
    print("  python3 leds.py green")
    print("  python3 leds.py blue")
    print("  python3 leds.py white")
    print("  python3 leds.py off")
    print("  python3 leds.py custom R G B")
    sys.exit(0)

cmd = sys.argv[1].lower()

if cmd == "red":
    set_color((255, 0, 0))

elif cmd == "green":
    set_color((0, 255, 0))

elif cmd == "blue":
    set_color((0, 0, 255))

elif cmd == "white":
    set_color((255, 255, 255))

elif cmd == "off":
    set_color((0, 0, 0))

elif cmd == "custom":
    if len(sys.argv) != 5:
        print("Usage: python3 leds.py custom R G B")
        sys.exit(1)
    r = int(sys.argv[2])
    g = int(sys.argv[3])
    b = int(sys.argv[4])
    set_color((r, g, b))

else:
    print("Unknown command.")
