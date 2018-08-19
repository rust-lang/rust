// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, RustcEncodable, RustcDecodable, Clone, Copy, Debug)]
pub enum Abi {
    // NB: This ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Single platform ABIs
    Cdecl,
    Stdcall,
    Fastcall,
    Vectorcall,
    Thiscall,
    Aapcs,
    Win64,
    SysV64,
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    AmdGpuKernel,

    // Multiplatform / generic ABIs
    Rust,
    C,
    System,
    RustIntrinsic,
    RustCall,
    PlatformIntrinsic,
    Unadjusted
}

#[derive(Copy, Clone)]
pub struct AbiData {
    abi: Abi,

    /// Name of this ABI as we like it called.
    name: &'static str,

    /// A generic ABI is supported on all platforms.
    generic: bool,
}

#[allow(non_upper_case_globals)]
const AbiDatas: &[AbiData] = &[
    // Platform-specific ABIs
    AbiData {abi: Abi::Cdecl, name: "cdecl", generic: false },
    AbiData {abi: Abi::Stdcall, name: "stdcall", generic: false },
    AbiData {abi: Abi::Fastcall, name: "fastcall", generic: false },
    AbiData {abi: Abi::Vectorcall, name: "vectorcall", generic: false},
    AbiData {abi: Abi::Thiscall, name: "thiscall", generic: false},
    AbiData {abi: Abi::Aapcs, name: "aapcs", generic: false },
    AbiData {abi: Abi::Win64, name: "win64", generic: false },
    AbiData {abi: Abi::SysV64, name: "sysv64", generic: false },
    AbiData {abi: Abi::PtxKernel, name: "ptx-kernel", generic: false },
    AbiData {abi: Abi::Msp430Interrupt, name: "msp430-interrupt", generic: false },
    AbiData {abi: Abi::X86Interrupt, name: "x86-interrupt", generic: false },
    AbiData {abi: Abi::AmdGpuKernel, name: "amdgpu-kernel", generic: false },

    // Cross-platform ABIs
    AbiData {abi: Abi::Rust, name: "Rust", generic: true },
    AbiData {abi: Abi::C, name: "C", generic: true },
    AbiData {abi: Abi::System, name: "system", generic: true },
    AbiData {abi: Abi::RustIntrinsic, name: "rust-intrinsic", generic: true },
    AbiData {abi: Abi::RustCall, name: "rust-call", generic: true },
    AbiData {abi: Abi::PlatformIntrinsic, name: "platform-intrinsic", generic: true },
    AbiData {abi: Abi::Unadjusted, name: "unadjusted", generic: true },
];

/// Returns the ABI with the given name (if any).
pub fn lookup(name: &str) -> Option<Abi> {
    AbiDatas.iter().find(|abi_data| name == abi_data.name).map(|&x| x.abi)
}

pub fn all_names() -> Vec<&'static str> {
    AbiDatas.iter().map(|d| d.name).collect()
}

impl Abi {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn data(self) -> &'static AbiData {
        &AbiDatas[self.index()]
    }

    pub fn name(self) -> &'static str {
        self.data().name
    }

    pub fn generic(self) -> bool {
        self.data().generic
    }
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\"{}\"", self.name())
    }
}

#[allow(non_snake_case)]
#[test]
fn lookup_Rust() {
    let abi = lookup("Rust");
    assert!(abi.is_some() && abi.unwrap().data().name == "Rust");
}

#[test]
fn lookup_cdecl() {
    let abi = lookup("cdecl");
    assert!(abi.is_some() && abi.unwrap().data().name == "cdecl");
}

#[test]
fn lookup_baz() {
    let abi = lookup("baz");
    assert!(abi.is_none());
}

#[test]
fn indices_are_correct() {
    for (i, abi_data) in AbiDatas.iter().enumerate() {
        assert_eq!(i, abi_data.abi.index());
    }
}
