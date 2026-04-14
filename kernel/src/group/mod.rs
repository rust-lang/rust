//! Group module: canonical coordination domain bridging the current Unix-shaped
//! process-group and session model toward the future `Group` ontology.
//!
//! # What Group is
//!
//! `Group` is the third canonical axis introduced by the phased migration:
//!
//! * **Task** — execution
//! * **Job**  — lifecycle
//! * **Group** — coordination (this module)
//!
//! A `Group` is an explicit, first-class coordination domain.  In v1 its only
//! observable property is its *role* (`GroupKind`), which distinguishes groups
//! that currently hold foreground terminal control from all other coordination
//! groupings.
//!
//! # Transitional mapping
//!
//! The current kernel still uses Unix-shaped process-group (`pgid`), session
//! (`sid`), and `session_leader` concepts stored in `Process`.  The TTY
//! foreground state lives in `ConsoleTtyState::foreground_pgid`.  None of
//! those structures are touched here.
//!
//! `kernel::group::bridge` is the **single translation point** from these
//! internal structures into the canonical `Group` type.
//!
//! # Future direction
//!
//! Once signal routing, TTY ownership, and session semantics are migrated into
//! `Group` terms (Phase 5+), this module will grow.  Until then, the
//! Unix-shaped internals remain operational and only their *meaning* is
//! replaced at system boundaries.

pub mod bridge;
