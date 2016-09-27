// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::convert;
use std::ffi;
use std::fmt;
use std::path;

/// Platform represented by a hyphen-delimited string. Does not
/// necessarily consist of three items. e.g.:
///
/// * `i686-apple-darwin`
/// * `powerpc-unknown-linux-gnu`
#[derive(Clone, Debug, Default, Eq, Hash, PartialEq, RustcDecodable)]
pub struct Triple(pub String);

impl Triple {
    pub fn arch(&self) -> &str {
        self.0.split('-').next().unwrap()
    }

    pub fn is_msvc(&self) -> bool {
        self.0.contains("msvc")
    }

    pub fn is_windows(&self) -> bool {
        self.0.contains("windows")
    }

    pub fn is_windows_msvc(&self) -> bool {
        self.0.contains("windows-msvc")
    }

    pub fn is_apple(&self) -> bool {
        self.0.contains("apple")
    }

    pub fn is_apple_darwin(&self) -> bool {
        self.0.contains("apple-darwin")
    }

    pub fn is_android(&self) -> bool {
        self.0.contains("android")
    }

    pub fn is_rumprun(&self) -> bool {
        self.0.contains("rumprun")
    }

    pub fn is_bitrig(&self) -> bool {
        self.0.contains("bitrig")
    }

    pub fn is_openbsd(&self) -> bool {
        self.0.contains("openbsd")
    }

    pub fn is_emscripten(&self) -> bool {
        self.0.contains("emscripten")
    }

    pub fn is_arm_linux_android(&self) -> bool {
        self.0.contains("arm-linux-android")
    }

    pub fn is_pc_windows_gnu(&self) -> bool {
        self.0.contains("pc-windows-gnu")
    }

    pub fn is_windows_gnu(&self) -> bool {
        self.0.contains("windows-gnu")
    }

    pub fn is_mips(&self) -> bool {
        self.0.contains("mips")
    }

    pub fn is_musl(&self) -> bool {
        self.0.contains("musl")
    }

    pub fn is_apple_ios(&self) -> bool {
        self.0.contains("musl")
    }

    pub fn is_i686(&self) -> bool {
        self.0.starts_with("i686")
    }
}

impl fmt::Display for Triple {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl convert::AsRef<path::Path> for Triple {
    fn as_ref(&self) -> &path::Path {
        self.0.as_ref()
    }
}

impl convert::AsRef<ffi::OsStr> for Triple {
    fn as_ref(&self) -> &ffi::OsStr {
        self.0.as_ref()
    }
}

impl<'a> convert::From<String> for Triple {
    fn from(s: String) -> Triple {
        Triple(s)
    }
}

impl<'a> convert::From<&'a str> for Triple {
    fn from(s: &'a str) -> Triple {
        Triple(s.into())
    }
}
