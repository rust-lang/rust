// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_id = "mach_triple#0.11-pre"]
#![license = "MIT/ASL2"]
#![crate_type = "rlib"]
#![crate_type = "dylib"]
#![doc(html_logo_url = "http://www.rust-lang.org/logos/rust-logo-128x128-blk-v2.png",
       html_favicon_url = "http://www.rust-lang.org/favicon.ico",
       html_root_url = "http://static.rust-lang.org/doc/master")]

extern crate syntax;

use std::result::{Result, Ok, Err};
use std::from_str::FromStr;
use std::{cmp, fmt, default};
use std::os::consts::{macos, freebsd, linux, android, win32};
use syntax::abi;

pub type KnownType<T> = Result<T, ~str>;
pub trait Known<T> {
    // Unwrap the value, or fails with a message using name.
    fn expect_known<'a>(&'a self, name: &str) -> &'a T;
}
impl<T> Known<T> for KnownType<T> {
    fn expect_known<'a>(&'a self, name: &str) -> &'a T {
        match self {
            &Ok(ref v) => v,
            &Err(ref v) => {
                fail!("Tried to unwrap unknown value `{}`. The unknown string was `{}`.",
                      name, v);
            }
        }
    }
}
pub trait ParseFromStr {
    fn parse_str(s: &str) -> KnownType<Self>;
}
impl<T: FromStr> ParseFromStr for T {
    fn parse_str(s: &str) -> KnownType<T> {
        let opt: Option<T> = FromStr::from_str(s);
        match opt {
            Some(v) => Ok(v),
            None    => Err(s.to_str()),
        }
    }
}

#[deriving(Hash, Clone, Eq, TotalEq)]
pub enum GnuType {
    GnuDefault,
    GnuEAbi,
    GnuEAbiHf,
    GnuX32,
}
impl FromStr for GnuType {
    fn from_str(s: &str) -> Option<GnuType> {
        match s {
            "gnu" => Some(GnuDefault),
            "gnueabi" => Some(GnuEAbi),
            "gnueabihf" => Some(GnuEAbiHf),
            "gnux32" => Some(GnuX32),
            _ => None,
        }
    }
}

#[deriving(Hash, Clone, Eq, TotalEq)]
pub enum Env {
    GnuEnv(GnuType),
    EAbiEnv,
    EAbiHfEnv,
    AndroidEnv,
    AndroidEAbiEnv,
    MsvcEnv,
    // here for completeness:
    ItaniumEnv,
}
impl FromStr for Env {
    fn from_str(s: &str) -> Option<Env> {
        let gnu_opt: Option<GnuType> = FromStr::from_str(s);
        match gnu_opt {
            Some(gnu) => Some(GnuEnv(gnu)),
            None => {
                match s {
                    "eabi"        => Some(EAbiEnv),
                    "eabihf"      => Some(EAbiHfEnv),
                    "android"     => Some(AndroidEnv),
                    "androideabi" => Some(AndroidEAbiEnv),
                    "msvc"        => Some(MsvcEnv),
                    "itanium"     => Some(ItaniumEnv),
                    _             => None,
                }
            }
        }
    }
}
#[deriving(Hash, Clone, Eq, TotalEq)]
pub enum Vendor {
    UnknownVendor,
    PCVendor,
    AppleVendor,
}
impl FromStr for Vendor {
    fn from_str(s: &str) -> Option<Vendor> {
        match s {
            "unknown" => Some(UnknownVendor),
            "pc" | "w64" => Some(PCVendor),
            "apple" => Some(AppleVendor),
            _ => None,
        }
    }
}

#[deriving(Hash, Clone, TotalEq)]
pub struct Triple {
    pub full: ~str,

    pub arch: abi::Architecture,
    pub os:   KnownType<abi::Os>,
    pub vendor: Option<KnownType<Vendor>>,
    pub env:  Option<KnownType<Env>>,
}
impl Triple {
    pub fn host_triple() -> Triple {
        // Get the host triple out of the build environment. This ensures that our
        // idea of the host triple is the same as for the set of libraries we've
        // actually built.  We can't just take LLVM's host triple because they
        // normalize all ix86 architectures to i386.
        //
        // Instead of grabbing the host triple (for the current host), we grab (at
        // compile time) the target triple that this rustc is built with and
        // calling that (at runtime) the host triple.
        FromStr::from_str(env!("CFG_COMPILER_HOST_TRIPLE")).unwrap()
    }
    pub fn expect_known_os(&self) -> abi::Os {
        self.os.expect_known("os").clone()
    }

    // Returns the corresponding (prefix, suffix) that files need to have for
    // dynamic libraries. Note this expects a known OS, which should all be
    // fine except for the session builder (the session builder won't proceed
    // if we don't have a known OS).
    pub fn dylibname(&self) -> (&'static str, &'static str) {
        match self.expect_known_os() {
            abi::OsWin32   => (win32::DLL_PREFIX,   win32::DLL_SUFFIX),
            abi::OsMacos   => (macos::DLL_PREFIX,   macos::DLL_SUFFIX),
            abi::OsLinux   => (linux::DLL_PREFIX,   linux::DLL_SUFFIX),
            abi::OsAndroid => (android::DLL_PREFIX, android::DLL_SUFFIX),
            abi::OsFreebsd => (freebsd::DLL_PREFIX, freebsd::DLL_SUFFIX),
        }
    }
}

impl FromStr for Triple {
    fn from_str(s: &str) -> Option<Triple> {
        // we require at least an arch.

        let splits: Vec<&str> = s.split_terminator('-').collect();
        if splits.len() < 2 {
            None
        } else {
            let splits = splits.as_slice();
            let arch = match FromStr::from_str(splits[0]) {
                Some(arch) => arch,
                None => return None,
            };
            if splits.len() == 2 {
                match splits[1] {
                    "mingw32msvc" => {
                        return Some(Triple {
                            full: s.to_str(),
                            arch: arch,
                            os:   Ok(abi::OsWin32),
                            vendor: None,
                            env:  Some(Ok(MsvcEnv)),
                        });
                    }
                    os => {
                        return Some(Triple {
                            full: s.to_str(),
                            arch: arch,
                            os:   ParseFromStr::parse_str(os),
                            vendor: None,
                            env:  None,
                        })
                    }
                }
            } else if splits.len() == 3 {
                match (splits[1], splits[2]) {
                    ("linux", "androideabi") => {
                        return Some(Triple {
                            full: s.to_str(),
                            arch: arch,
                            os: Ok(abi::OsAndroid),
                            vendor: None,
                            env: Some(Ok(AndroidEAbiEnv)),
                        })
                    }
                    _ => {
                        return Some(Triple {
                            full: s.to_str(),
                            arch: arch,
                            os: ParseFromStr::parse_str(splits[2]),
                            vendor: Some(ParseFromStr::parse_str(splits[1])),
                            env: None,
                        });
                    }
                }
            } else {
                Some(Triple {
                    full: s.to_str(),
                    arch: arch,
                    vendor: Some(ParseFromStr::parse_str(splits[1])),
                    os:     ParseFromStr::parse_str(splits[2]),
                    env:    Some(ParseFromStr::parse_str(splits[3])),
                })
            }
        }
    }
}
impl default::Default for Triple {
    fn default() -> Triple {
        Triple::host_triple()
    }
}
impl fmt::Show for Triple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.full.fmt(f)
    }
}
impl cmp::Eq for Triple {
    fn eq(&self, rhs: &Triple) -> bool {
        self.arch == rhs.arch &&
            self.os == rhs.os &&
            self.env == rhs.env
    }
}

#[cfg(test)]
mod test {
    use super::{Triple, Known,
                UnknownVendor,
                GnuEnv, GnuDefault,
                AndroidEAbiEnv, MsvcEnv};
    use syntax::abi;
    use std::to_str::ToStr;
    use std::from_str::FromStr;
    use std::fmt::Show;

    #[test]
    fn x86_64_unknown_linux_gnu() {
        let original = "x86_64-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86_64);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn i386_unknown_linux_gnu() {
        let original = "i386-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn i486_unknown_linux_gnu() {
        let original = "i486-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn i586_unknown_linux_gnu() {
        let original = "i586-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn i686_unknown_linux_gnu() {
        let original = "i686-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn i786_unknown_linux_gnu() {
        let original = "i786-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test] #[should_fail]
    fn unknownarch_unknown_linux_gnu() {
        let original = "unknownarch-unknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
    }
    #[test]
    fn x86_64_ununknown_linux_gnu() {
        // unknown vendor
        let original = "x86_64-ununknown-linux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86_64);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Err(~"ununknown")));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn x86_64_unknown_notlinux_gnu() {
        // unknown os
        let original = "x86_64-unknown-notlinux-gnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86_64);
        assert!(triple.os == Err(~"notlinux"));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Ok(GnuEnv(GnuDefault))));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn x86_64_unknown_linux_notgnu() {
        // unknown os
        let original = "x86_64-unknown-linux-notgnu";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86_64);
        assert!(triple.os == Ok(abi::OsLinux));
        assert!(triple.vendor == Some(Ok(UnknownVendor)));
        assert!(triple.env == Some(Err(~"notgnu")));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn i686_mingw32msvc() {
        // Odd one, this is.
        let original = "i686-mingw32msvc";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::X86);
        assert!(triple.os == Ok(abi::OsWin32));
        assert!(triple.vendor == None);
        assert!(triple.env == Some(Ok(MsvcEnv)));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test]
    fn arm_linux_androideabi() {
        // Another odd one, this one. Really should be arm-unknown-linux-androideabi or
        // maybe arm-android-linux-eabi.
        let original = "arm-linux-androideabi";
        let triple: Triple = FromStr::from_str(original).unwrap();
        assert!(triple.arch == abi::Arm);
        assert!(triple.os == Ok(abi::OsAndroid));
        assert!(triple.vendor == None);
        assert!(triple.env == Some(Ok(AndroidEAbiEnv)));
        assert_eq!(triple.to_str(), original.to_str());
    }
    #[test] #[should_fail]
    fn blank() {
        let original = "";
        let triple: Triple = FromStr::from_str(original).unwrap();
    }
    #[test] #[should_fail]
    fn blank_hyphen() {
        let original = "-";
        let triple: Triple = FromStr::from_str(original).unwrap();
    }
}
