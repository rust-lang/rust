trait Foo { fn f() -> int; }
trait Bar : Foo { fn g() -> int; }

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }

impl A : Bar {
    // Testing that this impl can call the impl of Foo
    fn g() -> int { self.f() }
}

fn main() {
    let a = &A { x: 3 };
    assert a.g() == 10;
}

