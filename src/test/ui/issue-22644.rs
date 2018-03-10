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
    let a : u32 = 0;
    let long_name : usize = 0;

    println!("{}", a as usize > long_name);
    println!("{}", a as usize < long_name); //~ ERROR `<` is interpreted as a start of generic
    println!("{}{}", a as usize < long_name, long_name);
    //~^ ERROR `<` is interpreted as a start of generic
    println!("{}", a as usize < 4); //~ ERROR `<` is interpreted as a start of generic
    println!("{}", a: usize > long_name);
    println!("{}{}", a: usize < long_name, long_name);
    //~^ ERROR `<` is interpreted as a start of generic
    println!("{}", a: usize < 4); //~ ERROR `<` is interpreted as a start of generic

    println!("{}", a
                   as
                   usize
                   < //~ ERROR `<` is interpreted as a start of generic
                   4);
    println!("{}", a


                   as


                   usize
                   < //~ ERROR `<` is interpreted as a start of generic
                   5);

    println!("{}", a as usize << long_name); //~ ERROR `<` is interpreted as a start of generic

    println!("{}", a: &mut 4); //~ ERROR expected type, found `4`
}
