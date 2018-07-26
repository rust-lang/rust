// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use io;

pub mod args;
pub mod env;
pub mod fs;
pub mod net;
#[path = "../../unix/path.rs"]
pub mod path;
pub mod pipe;
pub mod process;
pub mod os;

// This enum is used as the storage for a bunch of types which can't actually exist.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Void {}

pub fn unsupported<T>() -> io::Result<T> {
    Err(io::Error::new(
        io::ErrorKind::Other,
        "This function is not available on CloudABI.",
    ))
}
