// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #39828. If you make use of a module that
// consists only of generics, no code is generated, just a dummy
// module. The reduced graph consists of a single node (for that
// module) with no inputs. Since we only serialize edges, when we
// reload, we would consider that node dirty since it is not recreated
// (it is not the target of any edges).

// revisions:rpass1 rpass2
// aux-build:generic.rs

extern crate generic;
fn main() { }
