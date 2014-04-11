// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;
use std::from_str;

#[deriving(Eq, Hash, Clone, TotalEq)]
pub enum Os {
    OsWin32,
    OsMacos,
    OsLinux,
    OsAndroid,
    OsFreebsd,
}

impl from_str::FromStr for Os {
    fn from_str(s: &str) -> Option<Os> {
        match s {
            "mingw32" => Some(OsWin32),
            "win32"   => Some(OsWin32),
            "darwin"  => Some(OsMacos),
            "android" => Some(OsAndroid),
            "linux"   => Some(OsLinux),
            "freebsd" => Some(OsFreebsd),
            _ => None,
        }
    }
}

impl fmt::Show for Os {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &OsWin32   => "win32".fmt(f),
            &OsMacos   => "darwin".fmt(f),
            &OsLinux   => "linux".fmt(f),
            &OsAndroid => "android".fmt(f),
            &OsFreebsd => "freebsd".fmt(f),
        }
    }
}

#[allow(non_camel_case_types)]
#[deriving(Eq, Clone, Hash, TotalEq)]
pub enum Architecture {
    // NB. You cannot change the ordering of these
    // constants without adjusting IntelBits below.
    // (This is ensured by the test indices_are_correct().)
    X86,
    X86_64,
    Arm,
    Mips
}
impl from_str::FromStr for Architecture {
    fn from_str(s: &str) -> Option<Architecture> {
        match s {
            "i386" | "i486" | "i586" | "i686" | "i786" => Some(X86),
            "x86_64" => Some(X86_64),
            "arm" | "xscale" | "thumb" => Some(Arm),
            "mips" => Some(Mips),
            _ => None,
        }
    }
}

pub static IntelBits: u32 = (1 << (X86 as uint)) | (1 << (X86_64 as uint));
pub static ArmBits: u32 = (1 << (Arm as uint));

