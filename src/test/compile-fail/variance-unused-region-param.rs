// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we report an error for unused type parameters in types.

struct SomeStruct<'a> { x: u32 } //~ ERROR parameter `'a` is never used
enum SomeEnum<'a> { Nothing } //~ ERROR parameter `'a` is never used
trait SomeTrait<'a> { fn foo(&self); } //~ ERROR parameter `'a` is never used

fn main() {}
