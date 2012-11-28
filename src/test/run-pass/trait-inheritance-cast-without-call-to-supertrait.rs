// Testing that we can cast to a subtrait and call subtrait
// methods. Not testing supertrait methods

trait Foo {
    fn f() -> int;
}

trait Bar : Foo {
    fn g() -> int;
}

struct A {
    x: int
}

impl A : Foo {
    fn f() -> int { 10 }
}

impl A : Bar {
    fn g() -> int { 20 }
}

fn main() {
    let a = &A { x: 3 };
    let afoo = a as &Foo;
    let abar = a as &Bar;
    assert afoo.f() == 10;
    assert abar.g() == 20;
}

