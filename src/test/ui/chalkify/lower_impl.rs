// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

trait Foo { }

#[rustc_dump_program_clauses] //~ ERROR program clause dump
impl<T: 'static> Foo for T where T: Iterator<Item = i32> { }

trait Bar {
    type Assoc;
}

impl<T> Bar for T where T: Iterator<Item = i32> {
    #[rustc_dump_program_clauses] //~ ERROR program clause dump
    type Assoc = Vec<T>;
}

fn main() {
    println!("hello");
}
