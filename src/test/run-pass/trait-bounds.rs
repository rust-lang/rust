// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait connection {
    fn read() -> int;
}

trait connection_factory<C: connection> {
    fn create() -> C;
}

type my_connection = ();
type my_connection_factory = ();

impl (): connection {
    fn read() -> int { 43 }
}

impl my_connection_factory: connection_factory<my_connection> {
    fn create() -> my_connection { () }
}

fn main() {
    let factory = ();
    let connection = factory.create();
    let result = connection.read();
    assert result == 43;
}
