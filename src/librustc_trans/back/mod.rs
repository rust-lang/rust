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
