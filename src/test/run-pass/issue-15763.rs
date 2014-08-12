// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#[deriving(PartialEq, Show)]
struct Bar {
    x: int
}
impl Drop for Bar {
    fn drop(&mut self) {
        assert_eq!(self.x, 22);
    }
}

#[deriving(PartialEq, Show)]
struct Foo {
    x: Bar,
    a: int
}

fn foo() -> Result<Foo, int> {
    return Ok(Foo {
        x: Bar { x: 22 },
        a: return Err(32)
    });
}

fn baz() -> Result<Foo, int> {
    Ok(Foo {
        x: Bar { x: 22 },
        a: return Err(32)
    })
}

// explicit immediate return
fn aa() -> int {
    return 3;
}

// implicit immediate return
fn bb() -> int {
    3
}

// implicit outptr return
fn cc() -> Result<int, int> {
    Ok(3)
}

// explicit outptr return
fn dd() -> Result<int, int> {
    return Ok(3);
}

trait A {
    fn aaa(self) -> int {
        3
    }
    fn bbb(self) -> int {
        return 3;
    }
    fn ccc(self) -> Result<int, int> {
        Ok(3)
    }
    fn ddd(self) -> Result<int, int> {
        return Ok(3);
    }
}

impl A for int {}

fn main() {
    assert_eq!(foo(), Err(32));
    assert_eq!(baz(), Err(32));

    assert_eq!(aa(), 3);
    assert_eq!(bb(), 3);
    assert_eq!(cc().unwrap(), 3);
    assert_eq!(dd().unwrap(), 3);

    let i = box 32i as Box<A>;
    assert_eq!(i.aaa(), 3);
    let i = box 32i as Box<A>;
    assert_eq!(i.bbb(), 3);
    let i = box 32i as Box<A>;
    assert_eq!(i.ccc().unwrap(), 3);
    let i = box 32i as Box<A>;
    assert_eq!(i.ddd().unwrap(), 3);
}
