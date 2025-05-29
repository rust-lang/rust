#[expect(clippy::module_inception)]
mod config;
pub mod flags;
pub mod parsing;
pub mod target_selection;
#[cfg(test)]
mod tests;
pub mod toml;
pub mod types;

pub use config::*;
pub use target_selection::TargetSelection;
pub use toml::change_id::ChangeId;
pub use toml::common::*;
pub use toml::rust::LldMode;
pub use toml::target::Target;
pub use types::*;
