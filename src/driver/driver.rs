// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[no_core];
extern mod core(vers = "0.7-pre");

#[cfg(rustpkg)]
extern mod this(name = "rustpkg", vers = "0.7-pre");

#[cfg(fuzzer)]
extern mod this(name = "fuzzer", vers = "0.7-pre");

#[cfg(rustdoc)]
extern mod this(name = "rustdoc", vers = "0.7-pre");

#[cfg(rusti)]
extern mod this(name = "rusti", vers = "0.7-pre");

#[cfg(rust)]
extern mod this(name = "rust", vers = "0.7-pre");

#[cfg(rustc)]
extern mod this(name = "rustc", vers = "0.7-pre");

fn main() { this::main() }
