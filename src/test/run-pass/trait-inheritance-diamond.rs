// B and C both require A, so D does as well, twice, but that's just fine

trait A { fn a(&self) -> int; }
trait B: A { fn b(&self) -> int; }
trait C: A { fn c(&self) -> int; }
trait D: B C { fn d(&self) -> int; }

struct S { bogus: () }

impl S: A { fn a(&self) -> int { 10 } }
impl S: B { fn b(&self) -> int { 20 } }
impl S: C { fn c(&self) -> int { 30 } }
impl S: D { fn d(&self) -> int { 40 } }

fn f<T: D>(x: &T) {
    assert x.a() == 10;
    assert x.b() == 20;
    assert x.c() == 30;
    assert x.d() == 40;
}

fn main() {
    let value = &S { bogus: () };
    f(value);
}