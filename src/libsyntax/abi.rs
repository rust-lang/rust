// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

#[deriving(PartialEq)]
pub enum Os { OsWin32, OsMacos, OsLinux, OsAndroid, OsFreebsd, OsiOS,
              OsDragonfly }

#[deriving(PartialEq, Eq, Hash, Encodable, Decodable, Clone)]
pub enum Abi {
    // NB: This ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Single platform ABIs come first (`for_arch()` relies on this)
    Cdecl,
    Stdcall,
    Fastcall,
    Aapcs,
    Win64,

    // Multiplatform ABIs second
    Rust,
    C,
    System,
    RustIntrinsic,
    RustCall,
}

#[allow(non_camel_case_types)]
#[deriving(PartialEq)]
pub enum Architecture {
    // NB. You cannot change the ordering of these
    // constants without adjusting IntelBits below.
    // (This is ensured by the test indices_are_correct().)
    X86,
    X86_64,
    Arm,
    Mips,
    Mipsel
}

static IntelBits: u32 = (1 << (X86 as uint)) | (1 << (X86_64 as uint));
static ArmBits: u32 = (1 << (Arm as uint));

pub struct AbiData {
    abi: Abi,

    // Name of this ABI as we like it called.
    name: &'static str,

    // Is it specific to a platform? If so, which one?  Also, what is
    // the name that LLVM gives it (in case we disagree)
    abi_arch: AbiArchitecture
}

pub enum AbiArchitecture {
    /// Not a real ABI (e.g., intrinsic)
    RustArch,
    /// An ABI that specifies cross-platform defaults (e.g., "C")
    AllArch,
    /// Multiple architectures (bitset)
    Archs(u32)
}

static AbiDatas: &'static [AbiData] = &[
    // Platform-specific ABIs
    AbiData {abi: Cdecl, name: "cdecl", abi_arch: Archs(IntelBits)},
    AbiData {abi: Stdcall, name: "stdcall", abi_arch: Archs(IntelBits)},
    AbiData {abi: Fastcall, name:"fastcall", abi_arch: Archs(IntelBits)},
    AbiData {abi: Aapcs, name: "aapcs", abi_arch: Archs(ArmBits)},
    AbiData {abi: Win64, name: "win64",
             abi_arch: Archs(1 << (X86_64 as uint))},

    // Cross-platform ABIs
    //
    // NB: Do not adjust this ordering without
    // adjusting the indices below.
    AbiData {abi: Rust, name: "Rust", abi_arch: RustArch},
    AbiData {abi: C, name: "C", abi_arch: AllArch},
    AbiData {abi: System, name: "system", abi_arch: AllArch},
    AbiData {abi: RustIntrinsic, name: "rust-intrinsic", abi_arch: RustArch},
    AbiData {abi: RustCall, name: "rust-call", abi_arch: RustArch},
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
    pub fn index(&self) -> uint {
        *self as uint
    }

    #[inline]
    pub fn data(&self) -> &'static AbiData {
        &AbiDatas[self.index()]
    }

    pub fn name(&self) -> &'static str {
        self.data().name
    }

    pub fn for_target(&self, os: Os, arch: Architecture) -> Option<Abi> {
        // If this ABI isn't actually for the specified architecture, then we
        // short circuit early
        match self.data().abi_arch {
            Archs(a) if a & arch.bit() == 0 => return None,
            Archs(_) | RustArch | AllArch => {}
        }
        // Transform this ABI as appropriate for the requested os/arch
        // combination.
        Some(match (*self, os, arch) {
            (System, OsWin32, X86) => Stdcall,
            (System, _, _) => C,
            (me, _, _) => me,
        })
    }
}

impl Architecture {
    fn bit(&self) -> u32 {
        1 << (*self as uint)
    }
}

impl fmt::Show for Abi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "\"{}\"", self.name())
    }
}

impl fmt::Show for Os {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OsLinux => "linux".fmt(f),
            OsWin32 => "win32".fmt(f),
            OsMacos => "macos".fmt(f),
            OsiOS => "ios".fmt(f),
            OsAndroid => "android".fmt(f),
            OsFreebsd => "freebsd".fmt(f),
            OsDragonfly => "dragonfly".fmt(f)
        }
    }
}

#[allow(non_snake_case_functions)]
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

    let bits = 1 << (X86 as uint);
    let bits = bits | 1 << (X86_64 as uint);
    assert_eq!(IntelBits, bits);

    let bits = 1 << (Arm as uint);
    assert_eq!(ArmBits, bits);
}

#[test]
fn pick_uniplatform() {
    assert_eq!(Stdcall.for_target(OsLinux, X86), Some(Stdcall));
    assert_eq!(Stdcall.for_target(OsLinux, Arm), None);
    assert_eq!(System.for_target(OsLinux, X86), Some(C));
    assert_eq!(System.for_target(OsWin32, X86), Some(Stdcall));
    assert_eq!(System.for_target(OsWin32, X86_64), Some(C));
    assert_eq!(System.for_target(OsWin32, Arm), Some(C));
    assert_eq!(Stdcall.for_target(OsWin32, X86), Some(Stdcall));
    assert_eq!(Stdcall.for_target(OsWin32, X86_64), Some(Stdcall));
}
