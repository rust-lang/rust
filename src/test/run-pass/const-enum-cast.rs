// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum A { A1, A2 }
enum B { B1=0, B2=2 }

fn main () {
    static c1: int = A2 as int;
    static c2: int = B2 as int;
    static c3: float = A2 as float;
    static c4: float = B2 as float;
    let a1 = A2 as int;
    let a2 = B2 as int;
    let a3 = A2 as float;
    let a4 = B2 as float;
    fail_unless!(c1 == 1);
    fail_unless!(c2 == 2);
    fail_unless!(c3 == 1.0);
    fail_unless!(c4 == 2.0);
    fail_unless!(a1 == 1);
    fail_unless!(a2 == 2);
    fail_unless!(a3 == 1.0);
    fail_unless!(a4 == 2.0);
}
