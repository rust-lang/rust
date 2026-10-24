//@ compile-flags: -Znext-solver
//@ check-pass

#[derive(Default)]
struct Foo {
    x: i32,
}

fn main() {
    let mut xs = <[Foo; 1]>::default();
    xs[0].x = 1;
    (&mut xs[0]).x = 2;
}
