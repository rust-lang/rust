#![crate_type = "rlib"]
//@ revisions: current next
//@ edition: 2021
//@[current] known-bug: #155151
//@[current] check-fail
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

pub fn wut() -> impl Sized {
    struct Foo { x: u32 }

    if false {
        // `foo` has an opaque type, but this function knows that it's `Foo`.
        let foo = wut();
        let _closure = move || {
            let Foo { x } = foo;
            // `x` should have been captured, but under old-solver the compiler
            // thinks it's uninitialized here.
            let _y = x;
        };
    }

    Foo { x: 7 }
}
