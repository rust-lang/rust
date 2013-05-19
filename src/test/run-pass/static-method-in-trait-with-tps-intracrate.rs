// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Deserializer {
    fn read_int(&self) -> int;
}

trait Deserializable<D:Deserializer> {
    fn deserialize(d: &D) -> Self;
}

impl<D:Deserializer> Deserializable<D> for int {
    fn deserialize(d: &D) -> int {
        return d.read_int();
    }
}

struct FromThinAir { dummy: () }

impl Deserializer for FromThinAir {
    fn read_int(&self) -> int { 22 }
}

pub fn main() {
    let d = FromThinAir { dummy: () };
    let i: int = Deserializable::deserialize(&d);
    assert_eq!(i, 22);
}
