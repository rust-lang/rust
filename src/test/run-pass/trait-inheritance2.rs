trait Foo { fn f() -> int; }
trait Bar { fn g() -> int; }
trait Baz { fn h() -> int; }

trait Quux: Foo, Bar, Baz { }

struct A { x: int }

impl A : Foo { fn f() -> int { 10 } }
impl A : Bar { fn g() -> int { 20 } }
impl A : Baz { fn h() -> int { 30 } }
impl A : Quux;

fn f<T: Quux Foo Bar Baz>(a: &T) {
    assert a.f() == 10;
    assert a.g() == 20;
    assert a.h() == 30;
}

fn main() {
    let a = &A { x: 3 };
    f(a);
}

