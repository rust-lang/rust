//@ revisions: current next
//@ [next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ check-pass

// Regression test for <https://github.com/rust-lang/rust/issues/155151>.
// Used to say binding `x` isn't initialized

pub fn wut() -> impl Sized {
    struct Foo {
        x: u32,
    }

    if false {
        let foo = wut();
        let _closure = move || {
            let Foo { x } = foo;
            let _y = x;
        };
    }

    Foo { x: 7 }
}

fn main() {}
