use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
/// Whether a tool can be compiled, tested or neither
pub enum ToolState {
    /// The tool compiles successfully, but the test suite fails
    TestFail = 1,
    /// The tool compiles successfully and its test suite passes
    TestPass = 2,
    /// The tool can't even be compiled
    BuildFail = 0,
}

impl Default for ToolState {
    fn default() -> Self {
        // err on the safe side
        ToolState::BuildFail
    }
}
