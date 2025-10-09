//! The `wasm32-wasip3` target is the next in the chain of `wasm32-wasip1`, then
//! `wasm32-wasip2`, then WASIp3. The main feature of WASIp3 is native async
//! support in the component model itself.
//!
//! Like `wasm32-wasip2` this target produces a component by default. Support
//! for `wasm32-wasip3` is very early as of the time of this writing so
//! components produced will still import WASIp2 APIs, but that's ok since it's
//! all component-model-level imports anyway. Over time the imports of the
//! standard library will change to WASIp3.

use crate::spec::Target;

pub(crate) fn target() -> Target {
    // As of now WASIp3 is a lightly edited wasip2 target, so start with that
    // and this may grow over time as more features are supported.
    let mut target = super::wasm32_wasip2::target();
    target.llvm_target = "wasm32-wasip3".into();
    target.options.env = "p3".into();
    target
}
