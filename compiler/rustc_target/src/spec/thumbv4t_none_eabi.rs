//! Targets the ARMv4T, with code as `t32` code by default.
//!
//! Primarily of use for the GBA, but usable with other devices too.
//!
//! Please ping @Lokathor if changes are needed.
//!
//! **Important:** This target profile **does not** specify a linker script. You
//! just get the default link script when you build a binary for this target.
//! The default link script is very likely wrong, so you should use
//! `-Clink-arg=-Tmy_script.ld` to override that with a correct linker script.

use crate::spec::{cvs, FramePointer};
use crate::spec::{PanicStrategy, RelocModel, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv4t-none-eabi".into(),
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

            // extra args passed to the external assembler (assuming `arm-none-eabi-as`):
            // * activate t32/a32 interworking
            // * use arch ARMv4T
            // * use little-endian
            asm_args: cvs!["-mthumb-interwork", "-march=armv4t", "-mlittle-endian",],

            // minimum extra features, these cannot be disabled via -C
            // Also force-enable 32-bit atomics, which allows the use of atomic load/store only.
            // The resulting atomics are ABI incompatible with atomics backed by libatomic.
            features: "+soft-float,+strict-align,+atomics-32".into(),

            panic_strategy: PanicStrategy::Abort,
            relocation_model: RelocModel::Static,
            // suggested from thumb_base, rust-lang/rust#44993.
            emit_debug_gdb_scripts: false,
            frame_pointer: FramePointer::MayOmit,

            main_needs_argc_argv: false,

            // don't have atomic compare-and-swap
            atomic_cas: false,
            has_thumb_interworking: true,

            ..super::thumb_base::opts()
        },
    }
}
