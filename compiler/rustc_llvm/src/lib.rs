#![feature(nll)]
#![feature(static_nobundle)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]

// NOTE: This crate only exists to allow linking on mingw targets.

use libc::{c_char, size_t};
use std::cell::RefCell;
use std::slice;

#[repr(C)]
pub struct RustString {
    pub bytes: RefCell<Vec<u8>>,
}

impl RustString {
    pub fn len(&self) -> usize {
        self.bytes.borrow().len()
    }
}

/// Appending to a Rust string -- used by RawRustStringOstream.
#[no_mangle]
pub unsafe extern "C" fn LLVMRustStringWriteImpl(
    sr: &RustString,
    ptr: *const c_char,
    size: size_t,
) {
    let slice = slice::from_raw_parts(ptr as *const u8, size as usize);

    sr.bytes.borrow_mut().extend_from_slice(slice);
}

/// Initialize targets enabled by the build script via `cfg(llvm_component = "...")`.
/// N.B., this function can't be moved to `rustc_codegen_llvm` because of the `cfg`s.
pub fn initialize_available_targets() {
    macro_rules! init_target(
        ($cfg:meta, $($method:ident),*) => { {
            #[cfg($cfg)]
            fn init() {
                extern "C" {
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
    init_target!(
        llvm_component = "x86",
        LLVMInitializeX86TargetInfo,
        LLVMInitializeX86Target,
        LLVMInitializeX86TargetMC,
        LLVMInitializeX86AsmPrinter,
        LLVMInitializeX86AsmParser
    );
    init_target!(
        llvm_component = "arm",
        LLVMInitializeARMTargetInfo,
        LLVMInitializeARMTarget,
        LLVMInitializeARMTargetMC,
        LLVMInitializeARMAsmPrinter,
        LLVMInitializeARMAsmParser
    );
    init_target!(
        llvm_component = "aarch64",
        LLVMInitializeAArch64TargetInfo,
        LLVMInitializeAArch64Target,
        LLVMInitializeAArch64TargetMC,
        LLVMInitializeAArch64AsmPrinter,
        LLVMInitializeAArch64AsmParser
    );
    init_target!(
        llvm_component = "amdgpu",
        LLVMInitializeAMDGPUTargetInfo,
        LLVMInitializeAMDGPUTarget,
        LLVMInitializeAMDGPUTargetMC,
        LLVMInitializeAMDGPUAsmPrinter,
        LLVMInitializeAMDGPUAsmParser
    );
    init_target!(
        llvm_component = "avr",
        LLVMInitializeAVRTargetInfo,
        LLVMInitializeAVRTarget,
        LLVMInitializeAVRTargetMC,
        LLVMInitializeAVRAsmPrinter,
        LLVMInitializeAVRAsmParser
    );
    init_target!(
        llvm_component = "mips",
        LLVMInitializeMipsTargetInfo,
        LLVMInitializeMipsTarget,
        LLVMInitializeMipsTargetMC,
        LLVMInitializeMipsAsmPrinter,
        LLVMInitializeMipsAsmParser
    );
    init_target!(
        llvm_component = "powerpc",
        LLVMInitializePowerPCTargetInfo,
        LLVMInitializePowerPCTarget,
        LLVMInitializePowerPCTargetMC,
        LLVMInitializePowerPCAsmPrinter,
        LLVMInitializePowerPCAsmParser
    );
    init_target!(
        llvm_component = "systemz",
        LLVMInitializeSystemZTargetInfo,
        LLVMInitializeSystemZTarget,
        LLVMInitializeSystemZTargetMC,
        LLVMInitializeSystemZAsmPrinter,
        LLVMInitializeSystemZAsmParser
    );
    init_target!(
        llvm_component = "jsbackend",
        LLVMInitializeJSBackendTargetInfo,
        LLVMInitializeJSBackendTarget,
        LLVMInitializeJSBackendTargetMC
    );
    init_target!(
        llvm_component = "msp430",
        LLVMInitializeMSP430TargetInfo,
        LLVMInitializeMSP430Target,
        LLVMInitializeMSP430TargetMC,
        LLVMInitializeMSP430AsmPrinter
    );
    init_target!(
        all(llvm_component = "msp430", llvm_has_msp430_asm_parser),
        LLVMInitializeMSP430AsmParser
    );
    init_target!(
        llvm_component = "riscv",
        LLVMInitializeRISCVTargetInfo,
        LLVMInitializeRISCVTarget,
        LLVMInitializeRISCVTargetMC,
        LLVMInitializeRISCVAsmPrinter,
        LLVMInitializeRISCVAsmParser
    );
    init_target!(
        llvm_component = "sparc",
        LLVMInitializeSparcTargetInfo,
        LLVMInitializeSparcTarget,
        LLVMInitializeSparcTargetMC,
        LLVMInitializeSparcAsmPrinter,
        LLVMInitializeSparcAsmParser
    );
    init_target!(
        llvm_component = "nvptx",
        LLVMInitializeNVPTXTargetInfo,
        LLVMInitializeNVPTXTarget,
        LLVMInitializeNVPTXTargetMC,
        LLVMInitializeNVPTXAsmPrinter
    );
    init_target!(
        llvm_component = "hexagon",
        LLVMInitializeHexagonTargetInfo,
        LLVMInitializeHexagonTarget,
        LLVMInitializeHexagonTargetMC,
        LLVMInitializeHexagonAsmPrinter,
        LLVMInitializeHexagonAsmParser
    );
    init_target!(
        llvm_component = "webassembly",
        LLVMInitializeWebAssemblyTargetInfo,
        LLVMInitializeWebAssemblyTarget,
        LLVMInitializeWebAssemblyTargetMC,
        LLVMInitializeWebAssemblyAsmPrinter,
        LLVMInitializeWebAssemblyAsmParser
    );
}
