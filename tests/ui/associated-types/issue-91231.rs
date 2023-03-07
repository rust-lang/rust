// check-pass

#![feature(extern_types)]
#![allow(dead_code)]

extern {
    type Extern;
}

trait Trait {
    type Type;
}

#[inline]
fn f<'a>(_: <&'a Extern as Trait>::Type) where &'a Extern: Trait {}

fn main() {}
