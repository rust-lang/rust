//! Place module: canonical world-context bridging the current
//! `Process`-shaped cwd/namespace/root model toward the future `Place` ontology.
//!
//! # What Place is
//!
//! `Place` is the fifth canonical axis in the phased migration:
//!
//! * **Task**      — execution
//! * **Job**       — lifecycle
//! * **Group**     — coordination
//! * **Authority** — permission context
//! * **Place**     — world/visibility context (this module)
//!
//! A `Place` is the explicit, first-class answer to the question
//! *"in what world does this execution occur?"*.  In Phase 8 its observable
//! properties are `cwd` (working directory), `namespace` (VFS mount view
//! label), and `root` (effective filesystem root).
//!
//! # Note on Presence
//!
//! **Presence has not yet been introduced as a live execution/interaction
//! concept.**  This module (Phase 8) intentionally isolates world-context
//! (Place) first.  Terminal attachment, UI attachment, and embodied
//! person-in-place relationships will be modelled under `Presence` in a
//! future phase.  Any terminal/UI/console attachment state visible in the
//! current `Process` struct is therefore *not* bridged here — it remains
//! quarantined as legacy compatibility state until Presence is introduced.
//!
//! # Migration inventory — Process world/context responsibilities
//!
//! The table below inventories every field in the current `Process` struct that
//! answers "where is this happening?" questions and maps it to its intended
//! future canonical home.
//!
//! | `Process` field       | World-context role                    | Intended canonical home       | Status       |
//! |-----------------------|---------------------------------------|-------------------------------|--------------|
//! | `cwd`                 | Current working directory             | `Place::cwd`                  | **Bridged**  |
//! | `namespace`           | VFS mount-table view                  | `Place::namespace`            | Provisional  |
//! | *(no root field yet)* | Effective filesystem root             | `Place::root`                 | Not yet added|
//! | `env`                 | Inherited Unix environment blob       | **Legacy compat** (quarantine)| Provisional  |
//! | `argv` / `auxv`       | Spawn-time invocation context         | **Legacy compat** (quarantine)| Provisional  |
//! | `pgid` / `sid`        | Unix session/process-group            | `Group` (Phase 4 / 5)         | Provisional  |
//! | `session_leader`      | Unix TTY foreground ownership         | `Group::kind` (Phase 4)       | Provisional  |
//! | `fd_table`            | Open-file resource table              | Future resource authority     | Provisional  |
//! | `signals`             | Signal state                          | Future authority concern      | Provisional  |
//!
//! # Quarantined legacy compatibility fields
//!
//! The following `Process` fields are **transitional compatibility state**.
//! They implement necessary Unix behaviour but do not describe the canonical
//! world context.  New public-facing code must not treat them as architectural
//! truth; they must be accessed only through bridge/compat layers:
//!
//! * `Process::env` — inherited Unix process-local environment blob
//! * `Process::argv` / `Process::auxv` — spawn-time invocation context
//! * `Process::pgid` / `Process::sid` / `Process::session_leader` —
//!   Unix session and process-group state (→ `Group`)
//! * `Process::fd_table` — open-file descriptor table (→ future resource authority)
//! * `Process::signals` — per-process signal state (→ future authority concern)
//!
//! # Transitional mapping
//!
//! The current kernel still stores all world/context state inside `Process`
//! with no explicit Place-shaped substructure.  `kernel::place::bridge` is
//! the **single translation point** from the internal `ProcessSnapshot` into
//! the canonical `Place` type.
//!
//! # Future direction
//!
//! Once per-process namespace isolation and chroot support are added to
//! `Process`, this module will surface them through the bridge.  Until then
//! the Unix-shaped internal structures remain operational and only their
//! *meaning at system boundaries* is replaced by Place vocabulary.

pub mod bridge;
