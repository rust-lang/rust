// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Useful conditions

pub use std::path::Path;
pub use package_id::PkgId;
pub use std::rt::io::FileStat;
pub use std::rt::io::process::ProcessExit;

condition! {
    pub bad_path: (Path, ~str) -> Path;
}

condition! {
    pub bad_stat: (Path, ~str) -> FileStat;
}

condition! {
    pub bad_kind: (~str) -> ();
}

condition! {
    pub nonexistent_package: (PkgId, ~str) -> Path;
}

condition! {
    pub copy_failed: (Path, Path) -> ();
}

condition! {
    pub missing_pkg_files: (PkgId) -> ();
}

condition! {
    pub bad_pkg_id: (Path, ~str) -> PkgId;
}

condition! {
    pub no_rust_path: (~str) -> Path;
}

condition! {
    pub failed_to_create_temp_dir: (~str) -> Path;
}

condition! {
    pub git_checkout_failed: (~str, Path) -> ();
}

condition! {
    // str is output of applying the command (first component)
    // to the args (second component)
    pub command_failed: (~str, ~[~str], ProcessExit) -> ~str;
}
