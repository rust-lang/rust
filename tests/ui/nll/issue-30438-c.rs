// Simplified regression test for #30438, inspired by arielb1.

trait Trait { type Out; }

struct Test<'a> { s: &'a str }

fn silly<'y, 'z>(_s: &'y Test<'z>) -> &'y <Test<'z> as Trait>::Out where 'z: 'static {
    let x = Test { s: "this cannot last" };
    &x
    //~^ ERROR: cannot return reference to local variable `x`
}

impl<'b> Trait for Test<'b> { type Out = Test<'b>; }

fn main() {
    let orig = Test { s: "Hello World" };
    let r = silly(&orig);
    println!("{}", orig.s); // OK since `orig` is valid
    println!("{}", r.s); // Segfault (method does not return a sane value)
}
