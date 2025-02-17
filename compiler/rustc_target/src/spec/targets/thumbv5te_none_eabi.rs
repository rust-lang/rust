//! Targets the ARMv5TE, with code as `t32` code by default.

use crate::spec::{FloatAbi, FramePointer, Target, TargetMetadata, TargetOptions, base, cvs};

pub(crate) fn target() -> Target {
    Target {
        llvm_target: "thumbv5te-none-eabi".into(),
        metadata: TargetMetadata {
            description: Some("Thumb-mode Bare ARMv5TE".into()),
            tier: Some(3),
            host_tools: Some(false),
            std: Some(false),
        },
        pointer_width: 32,
        arch: "arm".into(),
        /* Data layout args are '-' separated:
         * little endian
         * stack is 64-bit aligned (EABI)
         * pointers are 32-bit
         * i64 must be 64-bit aligned (EABI)
         * mangle names with ELF style
         * native integers are 32-bit
         * All other elements are default
         */
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".into(),

        options: TargetOptions {
            abi: "eabi".into(),
            llvm_floatabi: Some(FloatAbi::Soft),
            // extra args passed to the external assembler (assuming `arm-none-eabi-as`):
            // * activate t32/a32 interworking
            // * use arch ARMv5TE
            // * use little-endian
            asm_args: cvs!["-mthumb-interwork", "-march=armv5te", "-mlittle-endian",],
            // minimum extra features, these cannot be disabled via -C
            // Also force-enable 32-bit atomics, which allows the use of atomic load/store only.
            // The resulting atomics are ABI incompatible with atomics backed by libatomic.
            features: "+soft-float,+strict-align,+atomics-32".into(),
            frame_pointer: FramePointer::MayOmit,
            main_needs_argc_argv: false,
            // don't have atomic compare-and-swap
            atomic_cas: false,
            has_thumb_interworking: true,

            ..base::thumb::opts()
        },
    }
}
