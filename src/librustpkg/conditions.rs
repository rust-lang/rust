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

pub use core::path::Path;
pub use package_id::PkgId;

condition! {
    bad_path: (super::Path, ~str) -> super::Path;
}

condition! {
    nonexistent_package: (super::PkgId, ~str) -> super::Path;
}

condition! {
    copy_failed: (super::Path, super::Path) -> ();
}

condition! {
    missing_pkg_files: (super::PkgId) -> ();
}

condition! {
    bad_pkg_id: (super::Path, ~str) -> super::PkgId;
}
