//! This target is a variant of `wasm32-unknown-unknown` which uses the bindgen
//! ABI instead of the normal ABI.
use super::{wasm32_unknown_unknown, Target};

pub fn target() -> Target {
    let mut target = wasm32_unknown_unknown::target();
    target.options.os = "bindgen".to_string();
    target
}
