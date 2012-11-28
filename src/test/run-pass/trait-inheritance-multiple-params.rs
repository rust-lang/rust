trait A { fn a(&self) -> int; }
trait B: A { fn b(&self) -> int; }
trait C: A { fn c(&self) -> int; }

struct S { bogus: () }

impl S: A { fn a(&self) -> int { 10 } }
impl S: B { fn b(&self) -> int { 20 } }
impl S: C { fn c(&self) -> int { 30 } }

// Multiple type params, multiple levels of inheritance
fn f<X: A, Y: B, Z: C>(x: &X, y: &Y, z: &Z) {
    assert x.a() == 10;
    assert y.a() == 10;
    assert y.b() == 20;
    assert z.a() == 10;
    assert z.c() == 30;
}

fn main() {
    let s = &S { bogus: () };
    f(s, s, s);
}