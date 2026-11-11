#![warn(clippy::definition_in_module_root)]

// #[path] to a named file — definitions should not trigger.
#[path = "custom/impl.rs"]
pub mod things;
