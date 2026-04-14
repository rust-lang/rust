//! Job module: canonical lifecycle container bridging the current `Process`
//! model toward the future `Job` ontology.
//!
//! # Transitional mapping
//!
//! | Canonical concept | Current internal backing  | Future direction             |
//! |-------------------|---------------------------|------------------------------|
//! | `thingos::job::Job` | kernel `Process` (partial) | hollowed out by further phases |
//!
//! The `Process` struct is retained as-is for now.  This module introduces
//! the public `Job` shape at the edges of the system, allowing the new
//! ontology to appear externally while the internal machinery is migrated
//! gradually (Phase 3 onwards).

pub mod bridge;
