// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// xfail-android

// compile-flags:-Z extra-debug-info
// debugger:run

#[allow(unused_variable)];

trait Trait {
    fn method(&self) -> int { 0 }
}

struct Struct {
    a: int,
    b: f64
}

impl Trait for Struct {}

// There is no real test here yet. Just make sure that it compiles without crashing.
fn main() {
    let stack_struct = Struct { a:0, b: 1.0 };
    let reference: &Trait = &stack_struct as &Trait;
    let managed: @Trait = @Struct { a:2, b: 3.0 } as @Trait;
    let unique: ~Trait = ~Struct { a:2, b: 3.0 } as ~Trait;
}
