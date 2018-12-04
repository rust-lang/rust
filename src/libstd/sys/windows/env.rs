// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod os {
    pub const FAMILY: &str = "windows";
    pub const OS: &str = "windows";
    pub const DLL_PREFIX: &str = "";
    pub const DLL_SUFFIX: &str = ".dll";
    pub const DLL_EXTENSION: &str = "dll";
    pub const EXE_SUFFIX: &str = ".exe";
    pub const EXE_EXTENSION: &str = "exe";
}
