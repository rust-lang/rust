//@ run-rustfix
#![allow(unused)]
fn a() => usize { 0 }
//~^ ERROR return types are denoted using `->`

fn b(): usize { 0 }
//~^ ERROR return types are denoted using `->`

fn bar(_: u32) {}

fn baz() -> *const dyn Fn(u32) { unimplemented!() }

fn foo() {
    match () {
        _ if baz() == &bar as &dyn Fn(u32) => (),
        () => (),
    }
}

fn main() {
    let foo = |a: bool| => bool { a };
    //~^ ERROR return types are denoted using `->`
    dbg!(foo(false));

    let bar = |a: bool|: bool { a };
    //~^ ERROR return types are denoted using `->`
    dbg!(bar(false));
}
