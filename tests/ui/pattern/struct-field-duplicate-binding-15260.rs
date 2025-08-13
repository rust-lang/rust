//! Regression test for https://github.com/rust-lang/rust/issues/15260

struct Foo {
    a: usize,
}

fn main() {
    let Foo {
        a: _,
        a: _
        //~^ ERROR field `a` bound multiple times in the pattern
    } = Foo { a: 29 };

    let Foo {
        a,
        a: _
        //~^ ERROR field `a` bound multiple times in the pattern
    } = Foo { a: 29 };

    let Foo {
        a,
        a: _,
        //~^ ERROR field `a` bound multiple times in the pattern
        a: x
        //~^ ERROR field `a` bound multiple times in the pattern
    } = Foo { a: 29 };
}
