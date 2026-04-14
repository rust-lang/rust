# Root

**Root** is the heart of ThingOS. It is the in-memory graph database service that runs inside the kernel.

## Role

Root maintains the **System Graph**, which is the single source of truth for the entire OS state. Everything in ThingOS is a node ("Thing") or an edge in this graph.

*   Processes are nodes.
*   Windows are nodes.
*   Hardware devices are nodes.
*   Files are nodes.
*   Configuration is nodes.

## Architecture

Root runs as a kernel thread (`root_main`). It processes a queue of `RootOp` messages from:
1.  **Kernel**: Internal kernel logic registering devices or updating state.
2.  **Userspace**: Processes sending requests via system calls (mediated by the kernel).

## Operations

Root supports operations such as:
*   `CreateNode`: Create a new Thing.
*   `Link`: Create an edge between Things.
*   `PropSet`/`PropGet`: Read/write properties on Things.
*   `Query`: Execute graph queries (traversals, filtering).
*   `Watch`: Subscribe to changes in the graph.

## The Graph

The graph is a directed property graph.
*   **Nodes (Things)**: Have a 64-bit ID (`ThingId`).
*   **Edges**: Directed connections with a label (Symbol).
*   **Properties**: Key-value pairs attached to nodes. Keys are Symbols.

## Root Service

The Root service is the *only* component that can directly modify the graph topology. All other components (kernel drivers, userspace apps) must send requests to Root.

This serialization ensures consistency and allows for powerful features like system-wide undo/time-travel (conceptually) and reactive queries.
