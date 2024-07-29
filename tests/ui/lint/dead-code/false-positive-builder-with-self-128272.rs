// Regression test for #128272
// Tests that we do not consider a struct
// dead code if it is constructed in a
// const function taking `&self`


//@ check-pass

#![deny(dead_code)]

pub struct Foo {
    _f: i32
}

impl Foo {
    // This function constructs Foo but in #128272
    // the compiler warned that Foo is never constructed
    pub const fn new(&self) -> Foo {
        Foo { _f: 5 }
    }
}

fn main() {
    static _X: Foo = _X.new();
}
