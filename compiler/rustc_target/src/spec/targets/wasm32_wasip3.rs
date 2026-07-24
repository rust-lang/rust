//! The `wasm32-wasip3` target is the next in the chain of `wasm32-wasip1`, then
//! `wasm32-wasip2`, then WASIp3. The main feature of WASIp3 is native async
//! support in the component model itself.
//!
//! Like `wasm32-wasip2` this target produces a component by default. Support
//! for `wasm32-wasip3` is very early as of the time of this writing so
//! components produced will still import WASIp2 APIs, but that's ok since it's
//! all component-model-level imports anyway. Over time the imports of the
//! standard library will change to WASIp3.

use crate::spec::{Cc, Env, LinkerFlavor, Target, add_link_args};

pub(crate) fn target() -> Target {
    // As of now WASIp3 is a lightly edited wasip2 target, so start with that
    // and this may grow over time as more features are supported.
    let mut target = super::wasm32_wasip2::target();
    target.llvm_target = "wasm32-wasip3".into();
    target.metadata = crate::spec::TargetMetadata {
        description: Some("WebAssembly".into()),
        tier: Some(3),
        host_tools: Some(false),
        std: Some(true),
    };
    target.options.env = Env::P3;

    // The `--cooperative-threading` flag to the linker dictates the ABI that's
    // being used on this target which is to store the stack pointer in a
    // component model intrinsic location, for example, rather than a wasm
    // global.
    //
    // Note that this is only specified for `Cc::No`, because when `clang` is
    // being used as a linker it'll already pass this.
    add_link_args(
        &mut target.pre_link_args,
        LinkerFlavor::WasmLld(Cc::No),
        &["--cooperative-threading"],
    );

    target
}
