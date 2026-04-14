# Sprout

**Sprout** is the **Init Process** (PID 1) of userspace.

## Role

Sprout is the first process launched by the kernel. Its job is to bootstrap the userspace environment.

## Responsibilities

1.  **Service Supervision**: Sprout launches and monitors essential system services:
    *   **Bristle** (Input)
    *   **Bloom** (Compositor)
    *   **Blossom** (UI Renderer)
    *   **Ingestd** (Asset Watcher)
2.  **Orchestration**: It ensures services start in the correct dependency order.
3.  **App Launching**: It may launch default user applications or a shell/launcher.
4.  **Graph Initialization**: Sprout populates the initial userspace portion of the System Graph, creating necessary folders and service nodes.

If Sprout dies, the system panics (or reboots).
