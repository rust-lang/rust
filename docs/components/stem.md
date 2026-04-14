# Stem

**Stem** is the userspace library for interacting with the ThingOS kernel and the Root graph.

## Role

Stem acts as the "libc" or "standard library" for ThingOS applications. It provides safe Rust wrappers around raw system calls.

## Responsibilities

*   **System Calls**: Invoking kernel functions (`syscall` instruction wrappers).
*   **Graph Access**: Helper functions to query and modify the Root graph.
    *   `stem::graph::create_node`
    *   `stem::graph::link`
    *   `stem::graph::prop_get`
*   **Process Management**: Spawning processes, exiting.
*   **IPC**: Managing ports and message passing.

## Usage

All userspace applications (like `sprout`, `bloom`, `blossom`) depend on `stem`.

```rust
use stem::prelude::*;

fn main() {
    let node = stem::graph::create_node("my_app").expect("Failed to create node");
    // ...
}
```

Stem ensures that applications play by the rules of the system, providing a safe and idiomatic interface to the raw capabilities of the kernel and Root.
