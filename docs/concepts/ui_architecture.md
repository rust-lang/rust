Canonical Layering (Contract)

Apps / Clients

- Own intent: “this is what my window looks like.”
- Emit DrawList generations into the graph.
- Never reason about pixels, damage, caching, or history.

Graph

- Owns truth: current DrawLists, assets, window topology.
- Is the only IPC path.
- Provides watches + snapshots, nothing semantic.

Bloom

- Owns work: turning vectors into pixels efficiently.
- Derives all secondary facts (bounds, damage, caches).
- Writes back only derived, optional hints (never required for correctness).

The litmus test

If you freeze the graph at time T and restart Bloom, the screen should
eventually look identical. If not, some layer is doing work it does not own.

DrawList: the published program

One DrawList = one immutable generation. New visual state means a new
generation. Old generations are disposable by Bloom.

DrawList node schema (graph)

- `window_id`
- `generation`
- `cmd_root` (edge to structured commands or chunk nodes)
- optional: `declared_assets` (fonts, images)

Rules

- No in-place mutation of commands.
- Bloom can diff by generation number, not content.

Damage ownership

- Apps never emit damage.
- The graph never stores damage as authoritative state.
- Bloom derives damage from generation changes, window movement/resize,
  z-order changes, and cursor movement.

Bloom hot loop (conceptual)

1. Drain graph watches.
2. For each affected window:
   - if generation changed: ingest new DrawList, compute bounds, invalidate caches.
3. Compute union damage.
4. Rasterize what intersects damage.
5. Composite window surfaces.
6. Present.

Ingest vs Render (Bloom internal split)

Ingest

- Reads the graph.
- Parses structured nodes.
- Builds a fast internal representation.
- Computes bounds once.

Render

- Consumes the internal representation.
- Does rasterization + composition.
- Knows nothing about the graph.

Optional graph writes

If Bloom writes derived information (damage rectangles, caches, snapshots),
they are debug/introspection/recovery hints only and must never be required
for correctness.

Deletion checklist (after refactor)

- Legacy snapshot paths.
- Double rasterization layers.
- App-side optimizations.
- Old bloom/blossom split leftovers.
