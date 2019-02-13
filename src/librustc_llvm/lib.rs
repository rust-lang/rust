#![deny(rust_2018_idioms)]
#![feature(nll)]
#![feature(static_nobundle)]

#![doc(html_root_url = "https://doc.rust-lang.org/nightly/")]

// See librustc_cratesio_shim/Cargo.toml for a comment explaining this.
#[allow(unused_extern_crates)]
extern crate rustc_cratesio_shim;

// NOTE: This crate only exists to allow linking on mingw targets.

/// Initialize targets enabled by the build script via `cfg(llvm_component = "...")`.
/// N.B., this function can't be moved to `rustc_codegen_llvm` because of the `cfg`s.
pub fn initialize_available_targets() {
    macro_rules! init_target(
        ($cfg:meta, $($method:ident),*) => { {
            #[cfg($cfg)]
            fn init() {
                extern {
                    $(fn $method();)*
                }
                unsafe {
                    $($method();)*
                }
            }
            #[cfg(not($cfg))]
            fn init() { }
            init();
        } }
    );
    init_target!(llvm_component = "x86",
                 LLVMInitializeX86TargetInfo,
                 LLVMInitializeX86Target,
                 LLVMInitializeX86TargetMC,
                 LLVMInitializeX86AsmPrinter,
                 LLVMInitializeX86AsmParser);
    init_target!(llvm_component = "arm",
                 LLVMInitializeARMTargetInfo,
                 LLVMInitializeARMTarget,
                 LLVMInitializeARMTargetMC,
                 LLVMInitializeARMAsmPrinter,
                 LLVMInitializeARMAsmParser);
    init_target!(llvm_component = "aarch64",
                 LLVMInitializeAArch64TargetInfo,
                 LLVMInitializeAArch64Target,
                 LLVMInitializeAArch64TargetMC,
                 LLVMInitializeAArch64AsmPrinter,
                 LLVMInitializeAArch64AsmParser);
    init_target!(llvm_component = "amdgpu",
                 LLVMInitializeAMDGPUTargetInfo,
                 LLVMInitializeAMDGPUTarget,
                 LLVMInitializeAMDGPUTargetMC,
                 LLVMInitializeAMDGPUAsmPrinter,
                 LLVMInitializeAMDGPUAsmParser);
    init_target!(llvm_component = "mips",
                 LLVMInitializeMipsTargetInfo,
                 LLVMInitializeMipsTarget,
                 LLVMInitializeMipsTargetMC,
                 LLVMInitializeMipsAsmPrinter,
                 LLVMInitializeMipsAsmParser);
    init_target!(llvm_component = "powerpc",
                 LLVMInitializePowerPCTargetInfo,
                 LLVMInitializePowerPCTarget,
                 LLVMInitializePowerPCTargetMC,
                 LLVMInitializePowerPCAsmPrinter,
                 LLVMInitializePowerPCAsmParser);
    init_target!(llvm_component = "systemz",
                 LLVMInitializeSystemZTargetInfo,
                 LLVMInitializeSystemZTarget,
                 LLVMInitializeSystemZTargetMC,
                 LLVMInitializeSystemZAsmPrinter,
                 LLVMInitializeSystemZAsmParser);
    init_target!(llvm_component = "jsbackend",
                 LLVMInitializeJSBackendTargetInfo,
                 LLVMInitializeJSBackendTarget,
                 LLVMInitializeJSBackendTargetMC);
    init_target!(llvm_component = "msp430",
                 LLVMInitializeMSP430TargetInfo,
                 LLVMInitializeMSP430Target,
                 LLVMInitializeMSP430TargetMC,
                 LLVMInitializeMSP430AsmPrinter);
    init_target!(llvm_component = "riscv",
                 LLVMInitializeRISCVTargetInfo,
                 LLVMInitializeRISCVTarget,
                 LLVMInitializeRISCVTargetMC,
                 LLVMInitializeRISCVAsmPrinter,
                 LLVMInitializeRISCVAsmParser);
    init_target!(llvm_component = "sparc",
                 LLVMInitializeSparcTargetInfo,
                 LLVMInitializeSparcTarget,
                 LLVMInitializeSparcTargetMC,
                 LLVMInitializeSparcAsmPrinter,
                 LLVMInitializeSparcAsmParser);
    init_target!(llvm_component = "nvptx",
                 LLVMInitializeNVPTXTargetInfo,
                 LLVMInitializeNVPTXTarget,
                 LLVMInitializeNVPTXTargetMC,
                 LLVMInitializeNVPTXAsmPrinter);
    init_target!(llvm_component = "hexagon",
                 LLVMInitializeHexagonTargetInfo,
                 LLVMInitializeHexagonTarget,
                 LLVMInitializeHexagonTargetMC,
                 LLVMInitializeHexagonAsmPrinter,
                 LLVMInitializeHexagonAsmParser);
    init_target!(llvm_component = "webassembly",
                 LLVMInitializeWebAssemblyTargetInfo,
                 LLVMInitializeWebAssemblyTarget,
                 LLVMInitializeWebAssemblyTargetMC,
                 LLVMInitializeWebAssemblyAsmPrinter);
}
