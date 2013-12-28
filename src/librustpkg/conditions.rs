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

pub use crate_id::CrateId;
pub use std::io::FileStat;
pub use std::io::process::ProcessExit;
pub use std::path::Path;

condition! {
    pub bad_path: (Path, ~str) -> Path;
}

condition! {
    pub nonexistent_package: (CrateId, ~str) -> Path;
}

condition! {
    pub missing_pkg_files: (CrateId) -> ();
}

condition! {
    pub bad_pkg_id: (Path, ~str) -> CrateId;
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
