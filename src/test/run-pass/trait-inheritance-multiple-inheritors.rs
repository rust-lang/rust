trait A { fn a(&self) -> int; }
trait B: A { fn b(&self) -> int; }
trait C: A { fn c(&self) -> int; }

struct S { bogus: () }

impl S: A { fn a(&self) -> int { 10 } }
impl S: B { fn b(&self) -> int { 20 } }
impl S: C { fn c(&self) -> int { 30 } }

// Both B and C inherit from A
fn f<T: B C>(x: &T) {
    assert x.a() == 10;
    assert x.b() == 20;
    assert x.c() == 30;
}

fn main() {
    f(&S { bogus: () })
}