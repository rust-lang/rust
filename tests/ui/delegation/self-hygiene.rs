#![feature(fn_delegation)]

macro_rules! emit_self { () => { self } }
//~^ ERROR expected value, found module `self`
//~| ERROR expected value, found module `self`

struct S;
impl S {
    fn method(self) {
        emit_self!();
    }
}

fn foo(arg: u8) {}
reuse foo as bar {
    emit_self!()
}

fn main() {}
