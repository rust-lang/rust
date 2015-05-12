// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::path::{self, Path, PathBuf};
use std::ffi::OsString;

// Unfortunately, on windows, gcc cannot accept paths of the form `\\?\C:\...`
// (a verbatim path). This form of path is generally pretty rare, but the
// implementation of `fs::canonicalize` currently generates paths of this form,
// meaning that we're going to be passing quite a few of these down to gcc.
//
// For now we just strip the "verbatim prefix" of `\\?\` from the path. This
// will probably lose information in some cases, but there's not a whole lot
// more we can do with a buggy gcc...
pub fn fix_windows_verbatim_for_gcc(p: &Path) -> PathBuf {
    if !cfg!(windows) {
        return p.to_path_buf()
    }
    let mut components = p.components();
    let prefix = match components.next() {
        Some(path::Component::Prefix(p)) => p,
        _ => return p.to_path_buf(),
    };
    let disk = match prefix.kind() {
        path::Prefix::VerbatimDisk(disk) => disk,
        _ => return p.to_path_buf(),
    };
    let mut base = OsString::from(format!("{}:", disk as char));
    base.push(components.as_path());
    PathBuf::from(base)
}
