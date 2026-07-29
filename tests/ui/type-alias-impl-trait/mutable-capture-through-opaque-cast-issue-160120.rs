//@ revisions: old next
//@ edition: 2021
//@[next] compile-flags: -Znext-solver=globally

#![feature(type_alias_impl_trait)]

#[derive(Copy, Clone)]
struct Foo((u32, u32));

fn main() {
    type T = impl Copy;
    let foo: T = Foo((2, 2));

    let _closure = || {
        //[old]~^ ERROR cannot borrow `foo.0` as mutable
        let Foo(ref mut _cdr) = foo;
        //[next]~^ ERROR cannot borrow `foo.0` as mutable
    };
}
