//! Targets the ARMv4T, with code as `t32` code by default.
//!
//! Primarily of use for the GBA, but usable with other devices too.
//!
//! Please ping @Lokathor if changes are needed.
//!
//! This target profile assumes that you have the ARM binutils in your path (specifically the linker, `arm-none-eabi-ld`). They can be obtained for free for all major OSes from the ARM developer's website, and they may also be available in your system's package manager. Unfortunately, the standard linker that Rust uses (`lld`) only supports as far back as `ARMv5TE`, so we must use the GNU `ld` linker.
//!
//! **Important:** This target profile **does not** specify a linker script. You just get the default link script when you build a binary for this target. The default link script is very likely wrong, so you should use `-Clink-arg=-Tmy_script.ld` to override that with a correct linker script.

use crate::spec::{LinkerFlavor, Target, TargetOptions};

pub fn target() -> Target {
    Target {
        llvm_target: "thumbv4t-none-eabi".to_string(),
        target_endian: "little".to_string(),
        pointer_width: 32,
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: "".to_string(),
        target_vendor: "".to_string(),
        arch: "arm".to_string(),
        /* Data layout args are '-' separated:
         * little endian
         * stack is 64-bit aligned (EABI)
         * pointers are 32-bit
         * i64 must be 64-bit aligned (EABI)
         * mangle names with ELF style
         * native integers are 32-bit
         * All other elements are default
         */
        data_layout: "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        options: TargetOptions {
            linker: Some("arm-none-eabi-ld".to_string()),
            linker_is_gnu: true,

            // extra args passed to the external assembler (assuming `arm-none-eabi-as`):
            // * activate t32/a32 interworking
            // * use arch ARMv4T
            // * use little-endian
            asm_args: vec![
                "-mthumb-interwork".to_string(),
                "-march=armv4t".to_string(),
                "-mlittle-endian".to_string(),
            ],

            // minimum extra features, these cannot be disabled via -C
            features: "+soft-float,+strict-align".to_string(),

            main_needs_argc_argv: false,

            // No thread-local storage (just use a static Cell)
            has_elf_tls: false,

            // don't have atomic compare-and-swap
            atomic_cas: false,
            has_thumb_interworking: true,

            ..super::thumb_base::opts()
        },
    }
}
