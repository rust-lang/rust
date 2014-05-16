// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-pretty-expanded

#![allow(unused_must_use, dead_code)]
#![feature(macro_rules)]

use std::io::MemWriter;

struct Foo<'a> {
    writer: &'a mut Writer,
    other: &'a str,
}

fn borrowing_writer_from_struct_and_formatting_struct_field(foo: Foo) {
    write!(foo.writer, "{}", foo.other);
}

fn main() {
    let mut w = MemWriter::new();
    write!(&mut w as &mut Writer, "");
    write!(&mut w, ""); // should coerce
    println!("ok");
}
