#![feature(type_alias_impl_trait)]
//@ revisions: current next
//@ edition: 2021
//@[current] known-bug: #155151
//@[current] check-fail
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

fn main() {
    struct Foo { x: u32 }

    type T = impl Sized;
    // `foo` has an opaque type, but this function knows that it's `Foo`.
    let foo: T = Foo { x: 7 };

    let _closure = move || {
        let Foo { x } = foo;
        // `x` should have been captured, but under old-solver the compiler
        // thinks it's uninitialized here.
        let _y = x;
    };
}
