// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Base: Base2 + Base3{
    fn foo(&self) -> String;
    fn foo1(&self) -> String;
    fn foo2(&self) -> String{
        "base foo2".to_string()
    }
}

trait Base2: Base3{
    fn baz(&self) -> String;
}

trait Base3{
    fn root(&self) -> String;
}

trait Super: Base{
    fn bar(&self) -> String;
}

struct X;

impl Base for X {
    fn foo(&self) -> String{
        "base foo".to_string()
    }
    fn foo1(&self) -> String{
        "base foo1".to_string()
    }

}

impl Base2 for X {
    fn baz(&self) -> String{
        "base2 baz".to_string()
    }
}

impl Base3 for X {
    fn root(&self) -> String{
        "base3 root".to_string()
    }
}

impl Super for X {
    fn bar(&self) -> String{
        "super bar".to_string()
    }
}

pub fn main() {
    let n = X;
    let s = &n as &Super;
    assert_eq!(s.bar(),"super bar".to_string());
    assert_eq!(s.foo(),"base foo".to_string());
    assert_eq!(s.foo1(),"base foo1".to_string());
    assert_eq!(s.foo2(),"base foo2".to_string());
    assert_eq!(s.baz(),"base2 baz".to_string());
    assert_eq!(s.root(),"base3 root".to_string());
}
