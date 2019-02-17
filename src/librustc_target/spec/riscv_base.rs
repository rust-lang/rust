use crate::spec::abi::Abi;

// All the calling conventions trigger an assertion(Unsupported calling
// convention) in llvm on RISCV
pub fn abi_blacklist() -> Vec<Abi> {
    vec![
        Abi::Cdecl,
        Abi::Stdcall,
        Abi::Fastcall,
        Abi::Vectorcall,
        Abi::Thiscall,
        Abi::Aapcs,
        Abi::Win64,
        Abi::SysV64,
        Abi::PtxKernel,
        Abi::Msp430Interrupt,
        Abi::X86Interrupt,
        Abi::AmdGpuKernel,
    ]
}
