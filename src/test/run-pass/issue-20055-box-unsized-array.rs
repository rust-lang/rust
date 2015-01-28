// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #2005: Check that boxed fixed-size arrays are properly
// accounted for (namely, only deallocated if they were actually
// created) when they appear as temporaries in unused arms of a match
// expression.

pub fn foo(box_1: fn () -> Box<[i8; 1]>,
           box_2: fn () -> Box<[i8; 20]>,
           box_3: fn () -> Box<[i8; 300]>,
           box_4: fn () -> Box<[i8; 4000]>,
            ) {
    println!("Hello World 1");
    let _: Box<[i8]> = match 3 {
        1 => box_1(),
        2 => box_2(),
        3 => box_3(),
        _ => box_4(),
    };
    println!("Hello World 2");
}

pub fn main() {
    fn box_1() -> Box<[i8; 1]> { Box::new( [1i8] ) }
    fn box_2() -> Box<[i8; 20]> { Box::new( [1i8; 20] ) }
    fn box_3() -> Box<[i8; 300]> { Box::new( [1i8; 300] ) }
    fn box_4() -> Box<[i8; 4000]> { Box::new( [1i8; 4000] ) }

    foo(box_1, box_2, box_3, box_4);
}
