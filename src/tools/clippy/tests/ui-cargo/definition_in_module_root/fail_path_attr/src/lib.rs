#![warn(clippy::definition_in_module_root)]

// #[path] to a mod.rs — definitions inside should still trigger.
#[path = "custom/mod.rs"]
pub mod things;
