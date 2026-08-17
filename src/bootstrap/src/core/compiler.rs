use std::hash::{Hash, Hasher};

use crate::Build;
use crate::core::config::TargetSelection;

/// A structure representing a Rust compiler.
///
/// Each compiler has a `stage` that it is associated with and a `host` that
/// corresponds to the platform the compiler runs on.
#[derive(Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub struct Compiler {
    pub(crate) stage: u32,
    pub(crate) host: TargetSelection,
    /// Indicates whether the compiler was forced to use a specific stage.
    /// This field is ignored in `Hash` and `PartialEq` implementations as only the `stage`
    /// and `host` fields are relevant for those.
    pub(crate) forced_compiler: bool,
}

impl Hash for Compiler {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.stage.hash(state);
        self.host.hash(state);
    }
}

impl PartialEq for Compiler {
    fn eq(&self, other: &Self) -> bool {
        self.stage == other.stage && self.host == other.host
    }
}

impl Compiler {
    pub(crate) fn new(stage: u32, host: TargetSelection) -> Self {
        Self { stage, host, forced_compiler: false }
    }

    pub(crate) fn forced_compiler(&mut self, forced_compiler: bool) {
        self.forced_compiler = forced_compiler;
    }

    /// Returns `true` if this is a snapshot compiler for `build`'s configuration
    pub(crate) fn is_snapshot(&self, build: &Build) -> bool {
        self.stage == 0 && self.host == build.host_target
    }

    /// Indicates whether the compiler was forced to use a specific stage.
    pub(crate) fn is_forced_compiler(&self) -> bool {
        self.forced_compiler
    }
}
