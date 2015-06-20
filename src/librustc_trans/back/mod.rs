// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use session::Session;

use std::path::{PathBuf, Path};

pub use rustc_back::abi;
pub use rustc_back::rpath;
pub use rustc_back::svh;
pub use rustc_back::target_strs;

pub mod archive;
pub mod linker;
pub mod link;
pub mod lto;
pub mod write;
pub mod msvc;

pub fn expect_nacl_cross_path(sess: &Session) -> PathBuf {
    use std::env;
    let cross_path = sess.opts.cg.cross_path.clone();
    match cross_path.or_else(|| env::var("NACL_SDK_ROOT").ok() ) {
        None => sess.fatal("need cross path (-C cross-path, or via NACL_SDK_ROOT) \
                            for this target"),
        Some(p) => Path::new(&p).to_path_buf(),
    }
}
#[cfg(not(target_os = "nacl"))]
pub fn pnacl_toolchain(sess: &Session) -> PathBuf {
    #[cfg(windows)]
    fn get_os_for_nacl_toolchain(_sess: &Session) -> String { "win".to_string() }
    #[cfg(target_os = "linux")]
    fn get_os_for_nacl_toolchain(_sess: &Session) -> String { "linux".to_string() }
    #[cfg(target_os = "macos")]
    fn get_os_for_nacl_toolchain(_sess: &Session) -> String { "mac".to_string() }
    #[cfg(all(not(windows),
              not(target_os = "linux"),
              not(target_os = "macos"),
              not(target_os = "nacl")))]
    fn get_os_for_nacl_toolchain(sess: &Session) -> ! {
        sess.fatal("NaCl/PNaCl toolchain unsupported on this OS (update this if that's changed)");
    }

    let mut tc = expect_nacl_cross_path(sess);
    tc.push("toolchain");
    tc.push(&format!("{}_pnacl", get_os_for_nacl_toolchain(sess)));
    tc
}
