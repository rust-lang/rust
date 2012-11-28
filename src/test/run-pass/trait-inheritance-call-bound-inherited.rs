trait Foo { fn f() -> int; }
trait Bar : Foo { fn g() -> int; }

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }
impl A : Bar { fn g() -> int { 20 } }

// Call a function on Foo, given a T: Bar
fn gg<T:Bar>(a: &T) -> int {
    a.f()
}

fn main() {
    let a = &A { x: 3 };
    assert gg(a) == 10;
}

