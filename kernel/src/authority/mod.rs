//! Authority module: canonical permission context bridging the current
//! `Process`-shaped credential model toward the future `Authority` ontology.
//!
//! # What Authority is
//!
//! `Authority` is the fourth canonical axis introduced by the phased migration:
//!
//! * **Task**      — execution
//! * **Job**       — lifecycle
//! * **Group**     — coordination
//! * **Authority** — permission context (this module)
//!
//! An `Authority` is the explicit, first-class answer to the question
//! *"under what power does this action occur?"*.  In Phase 7 its only
//! observable property is its `name` (derived from the running process/thread
//! name) and an initially empty `capabilities` list.
//!
//! # Transitional mapping
//!
//! The current kernel still stores all permission-bearing state inside
//! `Process` (pid, exec_path, etc.) with no explicit uid/gid or capability
//! mask.  `kernel::authority::bridge` is the **single translation point**
//! from the internal `ProcessSnapshot` into the canonical `Authority` type.
//!
//! # Future direction
//!
//! Once uid/gid-like fields, capability masks, or a service-account substructure
//! are added to `Process` (Phase 5 authority substructure), this module will
//! surface them through the bridge.  Until then the Unix-shaped internal
//! structures remain operational and only their *meaning at system boundaries*
//! is replaced.

pub mod bridge;
