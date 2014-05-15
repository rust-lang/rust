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
    fn foo(&self) -> StrBuf;
    fn foo1(&self) -> StrBuf;
    fn foo2(&self) -> StrBuf{
        "base foo2".to_strbuf()
    }
}

trait Base2: Base3{
    fn baz(&self) -> StrBuf;
}

trait Base3{
    fn root(&self) -> StrBuf;
}

trait Super: Base{
    fn bar(&self) -> StrBuf;
}

struct X;

impl Base for X {
    fn foo(&self) -> StrBuf{
        "base foo".to_strbuf()
    }
    fn foo1(&self) -> StrBuf{
        "base foo1".to_strbuf()
    }

}

impl Base2 for X {
    fn baz(&self) -> StrBuf{
        "base2 baz".to_strbuf()
    }
}

impl Base3 for X {
    fn root(&self) -> StrBuf{
        "base3 root".to_strbuf()
    }
}

impl Super for X {
    fn bar(&self) -> StrBuf{
        "super bar".to_strbuf()
    }
}

pub fn main() {
    let n = X;
    let s = &n as &Super;
    assert_eq!(s.bar(),"super bar".to_strbuf());
    assert_eq!(s.foo(),"base foo".to_strbuf());
    assert_eq!(s.foo1(),"base foo1".to_strbuf());
    assert_eq!(s.foo2(),"base foo2".to_strbuf());
    assert_eq!(s.baz(),"base2 baz".to_strbuf());
    assert_eq!(s.root(),"base3 root".to_strbuf());
}
