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

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum Os {
    Windows,
    Macos,
    Linux,
    Android,
    Freebsd,
    iOS,
    Dragonfly,
    Bitrig,
    Netbsd,
    Openbsd,
    NaCl,
    Haiku,
    Solaris,
}

#[derive(PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Clone, Copy, Debug)]
pub enum Abi {
    // NB: This ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Single platform ABIs
    Cdecl,
    Stdcall,
    Fastcall,
    Vectorcall,
    Aapcs,
    Win64,
    SysV64,

    // Multiplatform / generic ABIs
    Rust,
    C,
    System,
    RustIntrinsic,
    RustCall,
    PlatformIntrinsic,
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Architecture {
    X86,
    X86_64,
    Arm,
    Mips,
    Mipsel
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
const AbiDatas: &'static [AbiData] = &[
    // Platform-specific ABIs
    AbiData {abi: Abi::Cdecl, name: "cdecl", generic: false },
    AbiData {abi: Abi::Stdcall, name: "stdcall", generic: false },
    AbiData {abi: Abi::Fastcall, name: "fastcall", generic: false },
    AbiData {abi: Abi::Vectorcall, name: "vectorcall", generic: false},
    AbiData {abi: Abi::Aapcs, name: "aapcs", generic: false },
    AbiData {abi: Abi::Win64, name: "win64", generic: false },
    AbiData {abi: Abi::SysV64, name: "sysv64", generic: false },

    // Cross-platform ABIs
    AbiData {abi: Abi::Rust, name: "Rust", generic: true },
    AbiData {abi: Abi::C, name: "C", generic: true },
    AbiData {abi: Abi::System, name: "system", generic: true },
    AbiData {abi: Abi::RustIntrinsic, name: "rust-intrinsic", generic: true },
    AbiData {abi: Abi::RustCall, name: "rust-call", generic: true },
    AbiData {abi: Abi::PlatformIntrinsic, name: "platform-intrinsic", generic: true },
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
    pub fn index(&self) -> usize {
        *self as usize
    }

    #[inline]
    pub fn data(&self) -> &'static AbiData {
        &AbiDatas[self.index()]
    }

    pub fn name(&self) -> &'static str {
        self.data().name
    }

    pub fn generic(&self) -> bool {
        self.data().generic
    }
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\"{}\"", self.name())
    }
}

impl fmt::Display for Os {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Os::Linux => "linux".fmt(f),
            Os::Windows => "windows".fmt(f),
            Os::Macos => "macos".fmt(f),
            Os::iOS => "ios".fmt(f),
            Os::Android => "android".fmt(f),
            Os::Freebsd => "freebsd".fmt(f),
            Os::Dragonfly => "dragonfly".fmt(f),
            Os::Bitrig => "bitrig".fmt(f),
            Os::Netbsd => "netbsd".fmt(f),
            Os::Openbsd => "openbsd".fmt(f),
            Os::NaCl => "nacl".fmt(f),
            Os::Haiku => "haiku".fmt(f),
            Os::Solaris => "solaris".fmt(f),
        }
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
