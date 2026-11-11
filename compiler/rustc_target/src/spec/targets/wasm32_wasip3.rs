//! The `wasm32-wasip3` target is the next in the chain of `wasm32-wasip1`, then
//! `wasm32-wasip2`, then WASIp3. The main feature of WASIp3 is native async
//! support in the component model itself.
//!
//! Like `wasm32-wasip2` this target produces a component by default.

use crate::spec::{Cc, Env, LinkerFlavor, Target, add_link_args};

pub(crate) fn target() -> Target {
    // For now wasip3 is a lightly-edited wasip2 target.
    let mut target = super::wasm32_wasip2::target();
    target.llvm_target = "wasm32-wasip3".into();
    target.metadata = crate::spec::TargetMetadata {
        description: Some("WebAssembly".into()),
        tier: Some(2),
        host_tools: Some(false),
        std: Some(true),
    };
    target.options.env = Env::P3;

    add_link_args(
        &mut target.pre_link_args,
        LinkerFlavor::WasmLld(Cc::No),
        &[
            // The `--cooperative-threading` flag to the linker dictates the ABI
            // that's being used on this target which is to store the stack
            // pointer in a component model intrinsic location, for example,
            // rather than a wasm global.
            //
            // Note that this is only specified for `Cc::No`, because when
            // `clang` is being used as a linker it'll already pass this.
            "--cooperative-threading",
            // This is used as the wasi-libc-defined symbol here is required for
            // this target to function. The Rust compiler's symbol exports
            // otherwise don't know about this symbol so it's manually exported
            // here.
            //
            // Note that this additionally is only specified for `Cc::No`
            // because when `clang` is used the symbol exports happen naturally
            // and this isn't needed.
            "--export-if-defined=__wasm_task_hook",
        ],
    );

    target
}
