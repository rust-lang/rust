// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern mod std;

trait Serializer {
}

trait Serializable {
    fn serialize<S: Serializer>(s: S);
}

impl int: Serializable {
    fn serialize<S: Serializer>(_s: S) { }
}

struct F<A> { a: A }

impl<A: Copy Serializable> F<A>: Serializable {
    fn serialize<S: Serializer>(s: S) {
        self.a.serialize(move s);
    }
}

impl io::Writer: Serializer {
}

fn main() {
    do io::with_str_writer |wr| {
        let foo = F { a: 1 };
        foo.serialize(wr);

        let bar = F { a: F {a: 1 } };
        bar.serialize(wr);
    };
}
