// tidy-alphabetical-start
#![allow(internal_features)]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/nightly-rustc/")]
#![doc(rust_logo)]
#![feature(extern_types)]
#![feature(rustdoc_internals)]
// tidy-alphabetical-end

use std::cell::RefCell;
use std::{ptr, slice};

use libc::size_t;

unsafe extern "C" {
    /// Opaque type that allows C++ code to write bytes to a Rust-side buffer,
    /// in conjunction with `RawRustStringOstream`. Use this as `&RustString`
    /// (Rust) and `RustStringRef` (C++) in FFI signatures.
    pub type RustString;
}

impl RustString {
    pub fn build_byte_buffer(closure: impl FnOnce(&Self)) -> Vec<u8> {
        let buf = RustStringInner::default();
        closure(buf.as_opaque());
        buf.into_inner()
    }
}

/// Underlying implementation of [`RustString`].
///
/// Having two separate types makes it possible to use the opaque [`RustString`]
/// in FFI signatures without `improper_ctypes` warnings. This is a workaround
/// for the fact that there is no way to opt out of `improper_ctypes` when
/// _declaring_ a type (as opposed to using that type).
#[derive(Default)]
struct RustStringInner {
    bytes: RefCell<Vec<u8>>,
}

impl RustStringInner {
    fn as_opaque(&self) -> &RustString {
        let ptr: *const RustStringInner = ptr::from_ref(self);
        // We can't use `ptr::cast` here because extern types are `!Sized`.
        let ptr = ptr as *const RustString;
        unsafe { &*ptr }
    }

    fn from_opaque(opaque: &RustString) -> &Self {
        // SAFETY: A valid `&RustString` must have been created via `as_opaque`.
        let ptr: *const RustString = ptr::from_ref(opaque);
        let ptr: *const RustStringInner = ptr.cast();
        unsafe { &*ptr }
    }

    fn into_inner(self) -> Vec<u8> {
        self.bytes.into_inner()
    }
}

/// Appends the contents of a byte slice to a [`RustString`].
///
/// This function is implemented in `rustc_llvm` so that the C++ code in this
/// crate can link to it directly, without an implied link-time dependency on
/// `rustc_codegen_llvm`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn LLVMRustStringWriteImpl(
    buf: &RustString,
    slice_ptr: *const u8, // Same ABI as `*const c_char`
    slice_len: size_t,
) {
    let slice = unsafe { slice::from_raw_parts(slice_ptr, slice_len) };
    RustStringInner::from_opaque(buf).bytes.borrow_mut().extend_from_slice(slice);
}

/// Initialize targets enabled by the build script via `cfg(llvm_component = "...")`.
/// N.B., this function can't be moved to `rustc_codegen_llvm` because of the `cfg`s.
pub fn initialize_available_targets() {
    macro_rules! init_target(
        ($cfg:meta, $($method:ident),*) => { {
            #[cfg($cfg)]
            fn init() {
                unsafe extern "C" {
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
        llvm_component = "m68k",
        LLVMInitializeM68kTargetInfo,
        LLVMInitializeM68kTarget,
        LLVMInitializeM68kTargetMC,
        LLVMInitializeM68kAsmPrinter,
        LLVMInitializeM68kAsmParser
    );
    init_target!(
        llvm_component = "csky",
        LLVMInitializeCSKYTargetInfo,
        LLVMInitializeCSKYTarget,
        LLVMInitializeCSKYTargetMC,
        LLVMInitializeCSKYAsmPrinter,
        LLVMInitializeCSKYAsmParser
    );
    init_target!(
        llvm_component = "loongarch",
        LLVMInitializeLoongArchTargetInfo,
        LLVMInitializeLoongArchTarget,
        LLVMInitializeLoongArchTargetMC,
        LLVMInitializeLoongArchAsmPrinter,
        LLVMInitializeLoongArchAsmParser
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
        llvm_component = "msp430",
        LLVMInitializeMSP430TargetInfo,
        LLVMInitializeMSP430Target,
        LLVMInitializeMSP430TargetMC,
        LLVMInitializeMSP430AsmPrinter,
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
        llvm_component = "xtensa",
        LLVMInitializeXtensaTargetInfo,
        LLVMInitializeXtensaTarget,
        LLVMInitializeXtensaTargetMC,
        LLVMInitializeXtensaAsmPrinter,
        LLVMInitializeXtensaAsmParser
    );
    init_target!(
        llvm_component = "webassembly",
        LLVMInitializeWebAssemblyTargetInfo,
        LLVMInitializeWebAssemblyTarget,
        LLVMInitializeWebAssemblyTargetMC,
        LLVMInitializeWebAssemblyAsmPrinter,
        LLVMInitializeWebAssemblyAsmParser
    );
    init_target!(
        llvm_component = "bpf",
        LLVMInitializeBPFTargetInfo,
        LLVMInitializeBPFTarget,
        LLVMInitializeBPFTargetMC,
        LLVMInitializeBPFAsmPrinter,
        LLVMInitializeBPFAsmParser
    );
}
