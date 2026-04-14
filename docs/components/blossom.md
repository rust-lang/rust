# Blossom

**Blossom** is the **UI Paint Service** for ThingOS.

## Role

Blossom is a specialized service that turns high-level "intent" (layout trees, vector shapes, styling) into pixels.

## Why Separate from Bloom?

*   **Bloom** is optimized for *compositing* (moving large buffers of pixels, alpha blending).
*   **Blossom** is optimized for *rendering* (drawing curves, text, gradients, shadows).

By separating them, we keep the compositor fast and responsive. If the UI rendering is heavy (e.g., complex vector art), it doesn't stutter the mouse cursor or window movement, which is handled by Bloom.

## Functionality

*   **Vector Graphics**: Renders SVG-like paths and shapes.
*   **Text Layout**: Handles font rendering and text flow.
*   **UI Layout**: Computes the size and position of UI widgets based on constraints (flexbox-like).
*   **Paint**: Writes the resulting pixels into a shared memory buffer (bytespace) which Bloom then displays.

Applications describe *what* they want to look like (the Scene Graph), and Blossom handles *how* to draw it.
