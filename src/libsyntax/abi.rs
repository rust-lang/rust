// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::Os::*;
pub use self::Abi::*;
pub use self::Architecture::*;
pub use self::AbiArchitecture::*;

use std::fmt;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Os {
    OsWindows,
    OsMacos,
    OsLinux,
    OsAndroid,
    OsFreebsd,
    OsiOS,
    OsDragonfly,
    OsBitrig,
    OsNetbsd,
    OsOpenbsd,
    OsNaCl,
}

#[derive(PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Clone, Copy, Debug)]
pub enum Abi {
    // NB: This ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Single platform ABIs come first (`for_arch()` relies on this)
    Cdecl,
    Stdcall,
    Fastcall,
    Vectorcall,
    Aapcs,
    Win64,

    // Multiplatform ABIs second
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

    // Name of this ABI as we like it called.
    name: &'static str,
}

#[derive(Copy, Clone)]
pub enum AbiArchitecture {
    /// Not a real ABI (e.g., intrinsic)
    RustArch,
    /// An ABI that specifies cross-platform defaults (e.g., "C")
    AllArch,
    /// Multiple architectures (bitset)
    Archs(u32)
}

#[allow(non_upper_case_globals)]
const AbiDatas: &'static [AbiData] = &[
    // Platform-specific ABIs
    AbiData {abi: Cdecl, name: "cdecl" },
    AbiData {abi: Stdcall, name: "stdcall" },
    AbiData {abi: Fastcall, name: "fastcall" },
    AbiData {abi: Vectorcall, name: "vectorcall"},
    AbiData {abi: Aapcs, name: "aapcs" },
    AbiData {abi: Win64, name: "win64" },

    // Cross-platform ABIs
    //
    // NB: Do not adjust this ordering without
    // adjusting the indices below.
    AbiData {abi: Rust, name: "Rust" },
    AbiData {abi: C, name: "C" },
    AbiData {abi: System, name: "system" },
    AbiData {abi: RustIntrinsic, name: "rust-intrinsic" },
    AbiData {abi: RustCall, name: "rust-call" },
    AbiData {abi: PlatformIntrinsic, name: "platform-intrinsic" }
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
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\"{}\"", self.name())
    }
}

impl fmt::Display for Os {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OsLinux => "linux".fmt(f),
            OsWindows => "windows".fmt(f),
            OsMacos => "macos".fmt(f),
            OsiOS => "ios".fmt(f),
            OsAndroid => "android".fmt(f),
            OsFreebsd => "freebsd".fmt(f),
            OsDragonfly => "dragonfly".fmt(f),
            OsBitrig => "bitrig".fmt(f),
            OsNetbsd => "netbsd".fmt(f),
            OsOpenbsd => "openbsd".fmt(f),
            OsNaCl => "nacl".fmt(f),
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
