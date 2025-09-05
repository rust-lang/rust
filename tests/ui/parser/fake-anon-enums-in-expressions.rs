//@ build-pass
// Regression test for https://github.com/rust-lang/rust/issues/107461, where anon enum syntax
// proposed in https://github.com/rust-lang/rfcs/issues/294 being parsed in a naïve way fails with
// exisitng valid syntax for closures.

struct A;
struct B;

#[derive(Debug)]
#[allow(unused)]
struct MyStruct {
    x: f32,
}

fn main() {
    let x = |_: fn() -> A | B;
    let B = x(|| A);

    let y = 1.0;
    let closure = |f: fn(&f32) -> f32| MyStruct { x: f(&y) };
    println!("foo: {:?}", closure(|x| *x + 3.0));
}
