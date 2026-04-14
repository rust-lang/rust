# Bristle

**Bristle** is the **Input Aggregation Service** for ThingOS.

## Role

Bristle collects input events from various hardware drivers and presents a unified input stream to the rest of the system (primarily **Bloom**).

## Functionality

*   **Drivers**: Reads from PS/2, USB, or other input drivers.
*   **Normalization**: Converts hardware-specific scan codes or packets into standard ThingOS input events.
*   **Aggregation**: Merges keyboard and mouse streams.
*   **Broadcasting**: Writes input state to the Graph (or sends events via IPC) so that the active window manager (Bloom) can react.

## Graph Representation

Bristle may update nodes like `svc.Input` with properties representing the current mouse position (`pointer.x`, `pointer.y`) or key states, allowing for reactive input handling at the graph level.
