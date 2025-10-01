use rustc_macros::{Decodable, Encodable, HashStable};
use rustc_span::Symbol;

#[derive(Copy, Clone, Debug, Decodable, Encodable, HashStable)]
pub struct IntrinsicDef {
    pub name: Symbol,
    /// Whether the intrinsic has no meaningful body and all backends need to shim all calls to it.
    pub must_be_overridden: bool,
    /// Whether the intrinsic can be invoked from stable const fn
    pub const_stable: bool,
}
