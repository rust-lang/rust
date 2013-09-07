// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Context data structure used by rustpkg

use std::os;
use extra::workcache;

#[deriving(Clone)]
pub struct Ctx {
    // If use_rust_path_hack is true, rustpkg searches for sources
    // in *package* directories that are in the RUST_PATH (for example,
    // FOO/src/bar-0.1 instead of FOO). The flag doesn't affect where
    // rustpkg stores build artifacts.
    use_rust_path_hack: bool,
    // The root directory containing the Rust standard libraries
    sysroot_opt: Path
}

#[deriving(Clone)]
pub struct BuildCtx {
    // Context for workcache
    workcache_cx: workcache::Context,
    // Everything else
    cx: Ctx
}

impl BuildCtx {
    pub fn sysroot_opt(&self) -> Path {
        self.cx.sysroot_opt.clone()
    }

    pub fn sysroot_to_use(&self) -> Path {
        self.cx.sysroot_to_use()
    }
}

impl Ctx {
    pub fn sysroot_opt(&self) -> Path {
        self.sysroot_opt.clone()
    }
}

impl Ctx {
    /// Debugging
    pub fn sysroot_opt_str(&self) -> ~str {
        self.sysroot_opt.to_str()
    }

    // Hack so that rustpkg can run either out of a rustc target dir,
    // or the host dir
    pub fn sysroot_to_use(&self) -> Path {
        if !in_target(&self.sysroot_opt) {
            self.sysroot_opt.clone()
        }
        else {
            self.sysroot_opt.pop().pop().pop()
        }
  }
}

/// We assume that if ../../rustc exists, then we're running
/// rustpkg from a Rust target directory. This is part of a
/// kludgy hack used to adjust the sysroot.
pub fn in_target(sysroot_opt: &Path) -> bool {
    debug!("Checking whether %s is in target", sysroot_opt.to_str());
    os::path_is_dir(&sysroot_opt.pop().pop().push("rustc"))
}
