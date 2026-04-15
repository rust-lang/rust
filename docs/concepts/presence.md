# Presence: Deferred Embodiment Model

**Status:** Schema defined (deferred) — runtime integration pending  
**Phase:** Deferred (schema-first definition)  
**Related:** [`thingos.presence`](../../thingos/src/presence.rs), [`thingos.place`](../place.rs), [`thingos.group`](../group.rs)

---

## What Presence Is

`Presence` is the canonical **deferred embodiment model** in ThingOS.  It is
a structured description of an entity's situated existence relative to the
system's social/spatial model.

It answers questions like:

- present *where*? (`place`)
- present in *what social context*? (`group`)
- present *how*? (`mode`)
- present *as what kind of embodiment*? (`embodiment`)

Presence is defined here as a **schema and conceptual boundary**, not a live
runtime system.  Runtime embodiment integration is deliberately deferred until
Place and Group foundations are stable.

> Presence should be defined as a stable descriptive schema before it becomes
> a runtime driver of behaviour.

---

## Why Deferred?

The foundations beneath Presence are still settling:

- **Place** is defined but namespace isolation and chroot are not yet implemented.
- **Group** semantics are minimal — signal routing and session semantics are
  still being extracted from the Unix process model.
- **Messaging/inbox routing** is being established.
- **Authority** ownership is being inventoried and extracted.
- **Lifecycle and coordination models** are still in motion.

That makes Presence a poor candidate for immediate behavioural integration and
an excellent candidate for **schema-first definition**.

Defining Presence now as a schema:

- establishes vocabulary before implementation pressure distorts it;
- keeps future embodiment work anchored to generated types;
- avoids ad hoc embodiment flags and one-off "is here" fields later;
- creates a convergence point for future location/social/embodiment concepts;
- preserves momentum without coupling unstable runtime semantics too early.

---

## Schema Overview

```
kind thingos.presence.mode = enum {
  Active, Latent, Remote, Projected, Deferred, Unavailable,
}

kind thingos.presence.embodiment_kind = enum {
  Direct, Remote, Inherited, Symbolic, Deferred,
}

kind thingos.presence = struct {
  subject:     ref<thingos.person>,
  place:       option<ref<thingos.place>>,
  group:       option<ref<thingos.group>>,
  mode:        thingos.presence.mode,
  embodiment:  option<thingos.presence.embodiment_kind>,
  observed_at: option<u64>,
}
```

---

## Field Rationale

| Field          | Why it exists                                               |
|----------------|-------------------------------------------------------------|
| `subject`      | Identifies *what* entity is present.  Uses `ref<thingos.person>` for now; future phases may generalise to a broader entity type. |
| `place`        | Identifies *where* it is present.  Optional: unknown/inapplicable place is valid.  Does not require Place runtime semantics to be complete. |
| `group`        | Identifies the social/coordination context.  Optional: presence without a group anchor is valid.  Does not force Group coordination semantics to be final. |
| `mode`         | Describes the current quality/state of the presence (active, latent, remote, etc.). |
| `embodiment`   | Lightweight descriptor of *how* embodied the presence is.  Optional; not all presences have a meaningful embodiment descriptor at this phase. |
| `observed_at`  | Unix milliseconds when this record was last known valid.  Optional; presence records may be constructed without a known observation timestamp. |

---

## PresenceMode Values

| Variant       | Meaning                                                          |
|---------------|------------------------------------------------------------------|
| `Active`      | The entity is actively present and reachable.                    |
| `Latent`      | Present but quiescent; not actively interacting.                 |
| `Remote`      | Present through a remote or mediated connection.                 |
| `Projected`   | Present through a derived or mirrored association.               |
| `Deferred`    | Presence status not yet determined.                              |
| `Unavailable` | Entity cannot currently be reached or verified as present.       |

---

## EmbodimentKind Values

| Variant     | Meaning                                                            |
|-------------|--------------------------------------------------------------------|
| `Direct`    | Direct, first-person, locally embodied presence.                   |
| `Remote`    | Presence mediated through a remote channel or proxy.               |
| `Inherited` | Presence inherited from a parent or enclosing context.             |
| `Symbolic`  | A symbolic or representational placeholder presence.               |
| `Deferred`  | Embodiment explicitly deferred pending future integration.         |

---

## What Presence Is *Not* (Yet)

Presence does **not** yet mean:

- **Scheduling residency** — Presence is not a scheduler primitive.
- **Automatic group joins** — Presence does not trigger group membership.
- **Message-routing membership** — Presence does not create inbox or routing entries.
- **Authority/capability decisions** — Presence carries no permission semantics.
- **Live sensor certainty** — Presence is not a real-time occupancy signal.
- **Device/avatar/body synchronisation** — Presence is not a full embodiment
  framework.
- **Physical pose, topology, or movement** — No spatial model here.
- **UI or shell presentation** — Presence is not a display concept yet.
- **Liveness guarantee** — A `Presence` record may be stale.

---

## Relationship to Place and Group

Presence depends on Place and Group **conceptually**, but not **operationally**.

- A `Presence` may reference a `Place` without requiring Place runtime
  semantics (namespace isolation, chroot, etc.) to be complete.
- A `Presence` may reference a `Group` without forcing Group coordination
  behaviour (signal fanout, TTY ownership, session semantics) to be final.
- `Presence` should not become a hidden Place/Group integration layer.

Both `place` and `group` fields are **optional** at the schema level.  A
presence record with `place: None, group: None` and `mode: Deferred` is valid
and useful — it names a subject and defers everything else.

---

## Design Decisions

### 1. Subject is `ref<thingos.person>` for now

The subject of a Presence should ideally be a generic entity reference
(person, agent, job, etc.).  In v1, `ref<thingos.person>` is used as the
concrete type because `thingos.person` already exists in the schema.  The
canonical Rust type wraps this as an opaque `EntityRef([u8; 16])`, making it
easy to generalise to a broader entity type in a future phase without changing
the structural schema.

### 2. Presence is not singular

The schema does not force one-subject-one-presence.  A single subject may have
multiple `Presence` records (different places, different modes, projected
copies).  The schema makes no assumption about cardinality.

### 3. No authority semantics

Authority/capability concepts are explicitly excluded.  Presence may later
*interact* with authority (e.g., "is this entity present with sufficient
authority to access this resource?") but that interaction belongs at a higher
layer, not in the Presence schema itself.

### 4. observed_at is a raw `u64`

A richer timestamp type may be introduced once the system has a stable time
abstraction.  For now, Unix milliseconds as `u64` is simple, compatible with
generated types, and sufficient for deferred use.

---

## Generated Type Plumbing

The `presence.kind` file in `tools/kindc/kinds/` feeds into the `kindc`
schema compiler.  Running `just kindc-gen` regenerates
`tools/kindc/fixtures/generated/mod.rs` with updated Rust types.

The canonical module at `thingos/src/presence.rs` wraps the generated type
IDs (`KIND_ID_THINGOS_PRESENCE`, etc.) in handwritten idiomatic Rust types
that follow the same conventions as `place.rs`, `group.rs`, and `authority.rs`.

---

## Deferred Integration Points

These are likely future work items, **not** in scope for this phase:

1. **Place/Group views** — once Place and Group are stable, `Presence` should
   gain a view layer that projects which entities are present in a given place
   or group.
2. **Presence update/event flow** — presence changes (arrival, departure, mode
   changes) should eventually emit events through the messaging substrate.
3. **UI/session representation** — Presence should eventually inform shell and
   compositor-level session state (who is at the terminal, what is foregrounded).
4. **Authority interplay** — once authority model matures, Presence may
   participate in authority checks (e.g., "is this entity present with a role
   that permits this action?").
5. **Message routing membership** — once inbox routing is established, Presence
   may determine which entities receive certain broadcast messages.
6. **Embodiment refinement** — as device/avatar/body model matures, `EmbodimentKind`
   will be extended or replaced by a richer descriptor.
7. **General entity reference** — `subject` should eventually accept any entity
   type (job, service, agent, person), not just `ref<thingos.person>`.

---

## Risks and Anti-Patterns to Avoid

- **Turning Presence into a stealth runtime subsystem** — any PR that makes
  Presence drive scheduling, routing, or authority should be rejected until
  Place and Group are stable.
- **Coupling to unstable Place/Group semantics** — Presence references them
  by ThingId only; it does not embed their runtime state.
- **Overfitting to one embodiment story** — the `EmbodimentKind` enum is
  intentionally broad; don't collapse it to "has terminal / no terminal".
- **Sneaking authority/liveness semantics in** — "active presence" is a mode
  flag, not a capability grant or scheduling hint.
- **Forcing singularity** — do not add "at most one presence per subject"
  constraints at this layer.
