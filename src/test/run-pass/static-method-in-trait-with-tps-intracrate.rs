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
    fn read_int() -> int;
}

trait Deserializable<D: Deserializer> {
    static fn deserialize(d: &D) -> self;
}

impl<D: Deserializer> int: Deserializable<D> {
    static fn deserialize(d: &D) -> int {
        return d.read_int();
    }
}

struct FromThinAir { dummy: () }

impl FromThinAir: Deserializer {
    fn read_int() -> int { 22 }
}

fn main() {
    let d = FromThinAir { dummy: () };
    let i: int = deserialize(&d);
    assert i == 22;
}