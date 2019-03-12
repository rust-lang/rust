use crate::spec::{LinkerFlavor, PanicStrategy, Target, TargetOptions, TargetResult, abi::Abi}; 
// use crate::spec::abi::Abi;

pub fn target() -> TargetResult {
    Ok(Target {
        llvm_target: "xtensa-none-elf".to_string(),
        target_endian: "little".to_string(),
        target_pointer_width: "32".to_string(),
        target_c_int_width: "32".to_string(),
        data_layout: "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i64:64-f64:64-a:0:32-n32".to_string(),
        arch: "xtensa".to_string(),
        target_os: "none".to_string(),
        target_env: String::new(),
        target_vendor: String::new(),
        linker_flavor: LinkerFlavor::Gcc,

        options: TargetOptions {
            executables: true,

            // The LLVM backend currently can't generate object files. To
            // workaround this LLVM generates assembly files which then we feed
            // to gcc to get object files. For this reason we have a hard
            // dependency on this specific gcc.
            // asm_args: vec!["-mcpu=generic".to_string()],
            linker: Some("xtensa-esp32-elf-gcc".to_string()),
            no_integrated_as: true,

            max_atomic_width: Some(32),
            atomic_cas: true,

            // Because these devices have very little resources having an
            // unwinder is too onerous so we default to "abort" because the
            // "unwind" strategy is very rare.
            panic_strategy: PanicStrategy::Abort,

            // Similarly, one almost always never wants to use relocatable
            // code because of the extra costs it involves.
            relocation_model: "static".to_string(),

            // Right now we invoke an external assembler and this isn't
            // compatible with multiple codegen units, and plus we probably
            // don't want to invoke that many gcc instances.
            default_codegen_units: Some(1),

            // Since MSP430 doesn't meaningfully support faulting on illegal
            // instructions, LLVM generates a call to abort() function instead
            // of a trap instruction. Such calls are 4 bytes long, and that is
            // too much overhead for such small target.
            trap_unreachable: false,

            // See the thumb_base.rs file for an explanation of this value
            emit_debug_gdb_scripts: false,

            abi_blacklist: vec![
                Abi::Stdcall,
                Abi::Fastcall,
                Abi::Vectorcall,
                Abi::Thiscall,
                Abi::Win64,
                Abi::SysV64,
            ],

            .. Default::default( )
        }
    })
}
