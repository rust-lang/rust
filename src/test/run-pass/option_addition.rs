// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let foo: int = 1;
    let bar: int = 2;
    let foobar = foo + bar;

    let nope = None::<int> + None::<int>;
    let somefoo = Some(foo) + None::<int>;
    let somebar = None::<int> + Some(bar);
    let somefoobar = Some(foo) + Some(bar);

<<<<<<< HEAD
    match nope {
        None => (),
        Some(foo) => fail!("expected None, found %?", foo)
    }
    assert_eq!(foo, somefoo.get());
    assert_eq!(bar, somebar.get());
    assert_eq!(foobar, somefoobar.get());
}

fn optint(input: int) -> Option<int> {
    if input == 0 {
        return None;
    }
    else {
        return Some(input);
    }
=======
    assert_eq!(nope, None::<int>);
    assert_eq!(somefoo, None::<int>);
    assert_eq!(somebar, None::<int>);
    assert_eq!(foobar, somefoobar.unwrap());
>>>>>>> 72080954b9deb3a6a5f793d2fd1ef32c3d5acb5d
}
