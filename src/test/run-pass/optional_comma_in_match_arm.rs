// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let x = 1;

    match x {
        1 => loop { break; },
        2 => while true { break; },
        3 => if true { () },
        4 => if true { () } else { () },
        5 => match () { () => () },
        6 => { () },
        7 => unsafe { () },
        _ => (),
    }

    match x {
        1 => loop { break; }
        2 => while true { break; }
        3 => if true { () }
        4 => if true { () } else { () }
        5 => match () { () => () }
        6 => { () }
        7 => unsafe { () }
        _ => ()
    }

    let r: &i32 = &x;

    match r {
        // Absence of comma should not cause confusion between a pattern
        // and a bitwise and.
        &1 => if true { () } else { () }
        &2 => (),
        _ =>()
    }
}
