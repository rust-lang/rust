// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

enum Foo {
    A(i32),
    B
}

fn match_enum() {
    let mut foo = Foo::B;
    let p = &mut foo;
    let _ = match foo { //[mir]~ ERROR [E0503]
        Foo::B => 1, //[mir]~ ERROR [E0503]
        _ => 2,
        Foo::A(x) => x //[ast]~ ERROR [E0503]
                       //[mir]~^ ERROR [E0503]
    };
    drop(p);
}


fn main() {
    let mut x = 1;
    let r = &mut x;
    let _ = match x { //[mir]~ ERROR [E0503]
        x => x + 1, //[ast]~ ERROR [E0503]
                    //[mir]~^ ERROR [E0503]
        y => y + 2, //[ast]~ ERROR [E0503]
                    //[mir]~^ ERROR [E0503]
    };
    drop(r);
}
