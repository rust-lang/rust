// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::to_bytes;

#[deriving(Eq)]
pub enum Os { OsWin32, OsMacos, OsLinux, OsAndroid, OsFreebsd, }

#[deriving(Eq)]
pub enum Abi {
    // NB: This ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Single platform ABIs come first (`for_arch()` relies on this)
    Cdecl,
    Stdcall,
    Fastcall,
    Aapcs,

    // Multiplatform ABIs second
    Rust,
    C,
    System,
    RustIntrinsic,
}

#[deriving(Eq)]
pub enum Architecture {
    // NB. You cannot change the ordering of these
    // constants without adjusting IntelBits below.
    // (This is ensured by the test indices_are_correct().)
    X86,
    X86_64,
    Arm,
    Mips
}

static IntelBits: u32 = (1 << (X86 as uint)) | (1 << (X86_64 as uint));
static ArmBits: u32 = (1 << (Arm as uint));

struct AbiData {
    abi: Abi,

    // Name of this ABI as we like it called.
    name: &'static str,

    // Is it specific to a platform? If so, which one?  Also, what is
    // the name that LLVM gives it (in case we disagree)
    abi_arch: AbiArchitecture
}

enum AbiArchitecture {
    RustArch,   // Not a real ABI (e.g., intrinsic)
    AllArch,    // An ABI that specifies cross-platform defaults (e.g., "C")
    Archs(u32)  // Multiple architectures (bitset)
}

#[deriving(Clone, Eq, Encodable, Decodable)]
pub struct AbiSet {
    priv bits: u32   // each bit represents one of the abis below
}

static AbiDatas: &'static [AbiData] = &[
    // Platform-specific ABIs
    AbiData {abi: Cdecl, name: "cdecl", abi_arch: Archs(IntelBits)},
    AbiData {abi: Stdcall, name: "stdcall", abi_arch: Archs(IntelBits)},
    AbiData {abi: Fastcall, name:"fastcall", abi_arch: Archs(IntelBits)},
    AbiData {abi: Aapcs, name: "aapcs", abi_arch: Archs(ArmBits)},

    // Cross-platform ABIs
    //
    // NB: Do not adjust this ordering without
    // adjusting the indices below.
    AbiData {abi: Rust, name: "Rust", abi_arch: RustArch},
    AbiData {abi: C, name: "C", abi_arch: AllArch},
    AbiData {abi: System, name: "system", abi_arch: AllArch},
    AbiData {abi: RustIntrinsic, name: "rust-intrinsic", abi_arch: RustArch},
];

fn each_abi(op: &fn(abi: Abi) -> bool) -> bool {
    /*!
     *
     * Iterates through each of the defined ABIs.
     */

    AbiDatas.iter().advance(|abi_data| op(abi_data.abi))
}

pub fn lookup(name: &str) -> Option<Abi> {
    /*!
     *
     * Returns the ABI with the given name (if any).
     */

    let mut res = None;

    do each_abi |abi| {
        if name == abi.data().name {
            res = Some(abi);
            false
        } else {
            true
        }
    };
    res
}

pub fn all_names() -> ~[&'static str] {
    AbiDatas.map(|d| d.name)
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

    pub fn for_target(&self, os: Os, arch: Architecture) -> Abi {
        match (*self, os, arch) {
            (System, OsWin32, X86) => Stdcall,
            (System, _, _) => C,
            (me, _, _) => me,
        }
    }
}

impl Architecture {
    fn bit(&self) -> u32 {
        1 << (*self as u32)
    }
}

impl AbiSet {
    pub fn from(abi: Abi) -> AbiSet {
        AbiSet { bits: (1 << abi.index()) }
    }

    #[inline]
    pub fn Rust() -> AbiSet {
        AbiSet::from(Rust)
    }

    #[inline]
    pub fn C() -> AbiSet {
        AbiSet::from(C)
    }

    #[inline]
    pub fn Intrinsic() -> AbiSet {
        AbiSet::from(RustIntrinsic)
    }

    pub fn default() -> AbiSet {
        AbiSet::C()
    }

    pub fn empty() -> AbiSet {
        AbiSet { bits: 0 }
    }

    #[inline]
    pub fn is_rust(&self) -> bool {
        self.bits == 1 << Rust.index()
    }

    #[inline]
    pub fn is_c(&self) -> bool {
        self.bits == 1 << C.index()
    }

    #[inline]
    pub fn is_intrinsic(&self) -> bool {
        self.bits == 1 << RustIntrinsic.index()
    }

    pub fn contains(&self, abi: Abi) -> bool {
        (self.bits & (1 << abi.index())) != 0
    }

    pub fn subset_of(&self, other_abi_set: AbiSet) -> bool {
        (self.bits & other_abi_set.bits) == self.bits
    }

    pub fn add(&mut self, abi: Abi) {
        self.bits |= (1 << abi.index());
    }

    pub fn each(&self, op: &fn(abi: Abi) -> bool) -> bool {
        each_abi(|abi| !self.contains(abi) || op(abi))
    }

    pub fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub fn for_target(&self, os: Os, arch: Architecture) -> Option<Abi> {
        // NB---Single platform ABIs come first

        let mut res = None;

        do self.each |abi| {
            let data = abi.data();
            match data.abi_arch {
                Archs(a) if (a & arch.bit()) != 0 => { res = Some(abi); false }
                Archs(_) => { true }
                RustArch | AllArch => { res = Some(abi); false }
            }
        };

        res.map(|r| r.for_target(os, arch))
    }

    pub fn check_valid(&self) -> Option<(Abi, Abi)> {
        let mut abis = ~[];
        do self.each |abi| { abis.push(abi); true };

        for (i, abi) in abis.iter().enumerate() {
            let data = abi.data();
            for other_abi in abis.slice(0, i).iter() {
                let other_data = other_abi.data();
                debug!("abis=({:?},{:?}) datas=({:?},{:?})",
                       abi, data.abi_arch,
                       other_abi, other_data.abi_arch);
                match (&data.abi_arch, &other_data.abi_arch) {
                    (&AllArch, &AllArch) => {
                        // Two cross-architecture ABIs
                        return Some((*abi, *other_abi));
                    }
                    (_, &RustArch) |
                    (&RustArch, _) => {
                        // Cannot combine Rust or Rust-Intrinsic with
                        // anything else.
                        return Some((*abi, *other_abi));
                    }
                    (&Archs(is), &Archs(js)) if (is & js) != 0 => {
                        // Two ABIs for same architecture
                        return Some((*abi, *other_abi));
                    }
                    _ => {}
                }
            }
        }

        return None;
    }
}

impl to_bytes::IterBytes for Abi {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.index().iter_bytes(lsb0, f)
    }
}

impl to_bytes::IterBytes for AbiSet {
    fn iter_bytes(&self, lsb0: bool, f: to_bytes::Cb) -> bool {
        self.bits.iter_bytes(lsb0, f)
    }
}

impl ToStr for Abi {
    fn to_str(&self) -> ~str {
        self.data().name.to_str()
    }
}

impl ToStr for AbiSet {
    fn to_str(&self) -> ~str {
        let mut strs = ~[];
        do self.each |abi| {
            strs.push(abi.data().name);
            true
        };
        format!("\"{}\"", strs.connect(" "))
    }
}

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

#[cfg(test)]
fn cannot_combine(n: Abi, m: Abi) {
    let mut set = AbiSet::empty();
    set.add(n);
    set.add(m);
    match set.check_valid() {
        Some((a, b)) => {
            assert!((n == a && m == b) ||
                         (m == a && n == b));
        }
        None => {
            fail!("Invalid match not detected");
        }
    }
}

#[cfg(test)]
fn can_combine(n: Abi, m: Abi) {
    let mut set = AbiSet::empty();
    set.add(n);
    set.add(m);
    match set.check_valid() {
        Some((_, _)) => {
            fail!("Valid match declared invalid");
        }
        None => {}
    }
}

#[test]
fn cannot_combine_cdecl_and_stdcall() {
    cannot_combine(Cdecl, Stdcall);
}

#[test]
fn cannot_combine_c_and_rust() {
    cannot_combine(C, Rust);
}

#[test]
fn cannot_combine_rust_and_cdecl() {
    cannot_combine(Rust, Cdecl);
}

#[test]
fn cannot_combine_rust_intrinsic_and_cdecl() {
    cannot_combine(RustIntrinsic, Cdecl);
}

#[test]
fn can_combine_system_and_cdecl() {
    can_combine(System, Cdecl);
}

#[test]
fn can_combine_c_and_stdcall() {
    can_combine(C, Stdcall);
}

#[test]
fn can_combine_aapcs_and_stdcall() {
    can_combine(Aapcs, Stdcall);
}

#[test]
fn abi_to_str_stdcall_aaps() {
    let mut set = AbiSet::empty();
    set.add(Aapcs);
    set.add(Stdcall);
    assert!(set.to_str() == ~"\"stdcall aapcs\"");
}

#[test]
fn abi_to_str_c_aaps() {
    let mut set = AbiSet::empty();
    set.add(Aapcs);
    set.add(C);
    debug!("set = {}", set.to_str());
    assert!(set.to_str() == ~"\"aapcs C\"");
}

#[test]
fn abi_to_str_rust() {
    let mut set = AbiSet::empty();
    set.add(Rust);
    debug!("set = {}", set.to_str());
    assert!(set.to_str() == ~"\"Rust\"");
}

#[test]
fn indices_are_correct() {
    for (i, abi_data) in AbiDatas.iter().enumerate() {
        assert_eq!(i, abi_data.abi.index());
    }

    let bits = 1 << (X86 as u32);
    let bits = bits | 1 << (X86_64 as u32);
    assert_eq!(IntelBits, bits);

    let bits = 1 << (Arm as u32);
    assert_eq!(ArmBits, bits);
}

#[cfg(test)]
fn get_arch(abis: &[Abi], os: Os, arch: Architecture) -> Option<Abi> {
    let mut set = AbiSet::empty();
    for &abi in abis.iter() {
        set.add(abi);
    }
    set.for_target(os, arch)
}

#[test]
fn pick_multiplatform() {
    assert_eq!(get_arch([C, Cdecl], OsLinux, X86), Some(Cdecl));
    assert_eq!(get_arch([C, Cdecl], OsLinux, X86_64), Some(Cdecl));
    assert_eq!(get_arch([C, Cdecl], OsLinux, Arm), Some(C));
}

#[test]
fn pick_uniplatform() {
    assert_eq!(get_arch([Stdcall], OsLinux, X86), Some(Stdcall));
    assert_eq!(get_arch([Stdcall], OsLinux, Arm), None);
    assert_eq!(get_arch([System], OsLinux, X86), Some(C));
    assert_eq!(get_arch([System], OsWin32, X86), Some(Stdcall));
    assert_eq!(get_arch([System], OsWin32, X86_64), Some(C));
    assert_eq!(get_arch([System], OsWin32, Arm), Some(C));
    assert_eq!(get_arch([Stdcall], OsWin32, X86), Some(Stdcall));
    assert_eq!(get_arch([Stdcall], OsWin32, X86_64), Some(Stdcall));
}
