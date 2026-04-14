# Bloom

**Bloom** is the **Compositor and Window Manager** for ThingOS.

## Role

Bloom is responsible for:
*   **Compositing**: Taking the visual output of various applications and combining them into the final image sent to the screen.
*   **Window Management**: Managing the position, size, and stacking order of surfaces.
*   **Input Routing**: Receiving input events (from **Bristle**) and dispatching them to the correct application.

## Architecture

Bloom runs as a userspace service. It:
1.  **Owns the Framebuffer**: It is the only process allowed to write to the physical screen.
2.  **Uses the Namespace**: It binds to display devices and runtime/session state through the VFS namespace (`/dev`, `/services`, `/run`, `/session`).
3.  **Composes**: It reads the pixel buffers associated with those surfaces and blends them.
4.  **Reads Desktop State**: It treats `/session/desktop/{wallpaper,mode,background_color}` as the source of truth for the background layer.

## Bring-up Rule

Bloom must be able to reach first paint without any Root graph, `ThingId`, or `UI_CROWN` bootstrap. Legacy graph-backed experiments are optional only and must not be part of the compositor's critical startup path.

## Interaction with Blossom

Bloom handles the *surfaces*, but it often delegates the *rendering* of complex vector UI to **Blossom**. However, Bloom itself is the final authority on what pixels go to the hardware.
