use crate::spec::abi::Abi;

// All the calling conventions trigger an assertion(Unsupported calling convention) in llvm on arm
pub fn unsupported_abis() -> Vec<Abi> {
    vec![
        Abi::Stdcall { unwind: false },
        Abi::Stdcall { unwind: true },
        Abi::Fastcall,
        Abi::Vectorcall,
        Abi::Thiscall { unwind: false },
        Abi::Thiscall { unwind: true },
        Abi::Win64,
        Abi::SysV64,
    ]
}
