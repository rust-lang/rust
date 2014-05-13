// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*

#5008 cast to &Trait causes code to segfault on method call

It fixes itself if the &Trait is changed to @Trait.
*/

trait Debuggable {
    fn debug_name(&self) -> StrBuf;
}

#[deriving(Clone)]
struct Thing {
    name: StrBuf,
}

impl Thing {
    fn new() -> Thing { Thing { name: "dummy".to_strbuf() } }
}

impl Debuggable for Thing {
    fn debug_name(&self) -> StrBuf { self.name.clone() }
}

fn print_name(x: &Debuggable)
{
    println!("debug_name = {}", x.debug_name());
}

pub fn main() {
    let thing = Thing::new();
    print_name(&thing as &Debuggable);
}
