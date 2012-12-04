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
extern mod core(vers = "0.5");

#[cfg(cargo)]
extern mod self(name = "cargo", vers = "0.5");

#[cfg(fuzzer)]
extern mod self(name = "fuzzer", vers = "0.5");

#[cfg(rustdoc)]
extern mod self(name = "rustdoc", vers = "0.5");

#[cfg(rusti)]
extern mod self(name = "rusti", vers = "0.5");

#[cfg(rustc)]
extern mod self(name = "rustc", vers = "0.5");

fn main() { self::main() }