//! NB: This target is in the process of being renamed to
//! `wasm32-wasip1`. For more information see:
//!
//! * <https://github.com/rust-lang/compiler-team/issues/607>
//! * <https://github.com/rust-lang/compiler-team/issues/695>

use crate::spec::Target;

pub fn target() -> Target {
    super::wasm32_wasip1::target()
}
