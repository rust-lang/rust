#[cfg(feature = "nightly")]
use rustc_macros::{Decodable_NoContext, Encodable_NoContext, StableHash};

// How to perform collapse macros debug info
/// if-ext - if macro from different crate (related to callsite code)
/// | cmd \ attr    | no  | (unspecified) | external | yes |
/// | no            | no  | no            | no       | no  |
/// | (unspecified) | no  | no            | if-ext   | yes |
/// | external      | no  | if-ext        | if-ext   | yes |
/// | yes           | yes | yes           | yes      | yes |
#[derive(Copy, Clone, Debug, Hash, PartialEq)]
#[cfg_attr(feature = "nightly", derive(StableHash, Encodable_NoContext, Decodable_NoContext))]
pub enum CollapseMacroDebuginfo {
    /// Don't collapse debuginfo for the macro
    No = 0,
    /// Unspecified value
    Unspecified = 1,
    /// Collapse debuginfo if the macro comes from a different crate
    External = 2,
    /// Collapse debuginfo for the macro
    Yes = 3,
}
