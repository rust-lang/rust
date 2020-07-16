//! Targets the Nintendo Game Boy Advance (GBA), a handheld game device from 2001.
//!
//! Please ping @Lokathor if changes are needed.
//!
//! Important: This target **does not** specify a linker script or the ROM
//! header. You'll still need to provide these yourself to construct a final
//! binary. Generally you'd do this with something like
//! `-Clink-arg=-Tmy_script.ld` and `-Clink-arg=my_crt.o`.

use crate::spec::{LinkerFlavor, LldFlavor, Target, TargetOptions, TargetResult};

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "thumbv4t-none-eabi".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        target_os: "none".to_string(),
        target_env: "gba".to_string(),
        target_vendor: "nintendo".to_string(),
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
        data_layout: "e-S64-p:32:32-i64:64-m:e-n32".to_string(),
        linker_flavor: LinkerFlavor::Ld,
        options: TargetOptions {
            linker: Some("arm-none-eabi-ld".to_string()),
            linker_is_gnu: true,

            // extra args passed to the external assembler
            asm_args: vec!["-mcpu=arm7tdmi".to_string(), "-mthumb-interwork".to_string()],

            cpu: "arm7tdmi".to_string(),

            // minimum extra features, these cannot be disabled via -C
            features: "+soft-float,+strict-align".to_string(),

            executables: true,

            relocation_model: RelocModel::Static,

            //function_sections: bool,
            //exe_suffix: String,
            main_needs_argc_argv: false,

            // if we have thread-local storage
            has_elf_tls: false,

            // don't have atomic compare-and-swap
            atomic_cas: false,

            // always just abort
            panic_strategy: PanicStrategy::Abort,

            // ABIs to not use
            unsupported_abis: super::arm_base::unsupported_abis(),

            // The minimum alignment for global symbols.
            min_global_align: Some(4),

            // no threads here
            singlethread: true,

            // GBA has no builtins
            no_builtins: true,

            // this is off just like in the `thumb_base`
            emit_debug_gdb_scripts: false,

            ..TargetOptions::default()
        },
    })
}
