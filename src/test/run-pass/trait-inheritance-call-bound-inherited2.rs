trait Foo { fn f() -> int; }
trait Bar : Foo { fn g() -> int; }
trait Baz : Bar { fn h() -> int; }

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }
impl A : Bar { fn g() -> int { 20 } }
impl A : Baz { fn h() -> int { 30 } }

// Call a function on Foo, given a T: Baz,
// which is inherited via Bar
fn gg<T: Baz>(a: &T) -> int {
    a.f()
}

fn main() {
    let a = &A { x: 3 };
    assert gg(a) == 10;
}

