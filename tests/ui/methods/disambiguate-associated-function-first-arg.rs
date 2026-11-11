struct A {}

fn main() {
    let _a = A {};
    _a.new(1);
    //~^ ERROR no method named `new` found for struct `A` in the current scope
}

trait M {
    fn new(_a: i32);
}
impl M for A {
    fn new(_a: i32) {}
}

trait N {
    fn new(_a: Self, _b: i32);
}
impl N for A {
    fn new(_a: Self, _b: i32) {}
}

trait O {
    fn new(_a: Self, _b: i32);
}
impl O for A {
    fn new(_a: A, _b: i32) {}
}

struct S;

trait TraitA {
    fn f(self);
}
trait TraitB {
    fn f(self);
}

impl<T> TraitA for T {
    fn f(self) {}
}
impl<T> TraitB for T {
    fn f(self) {}
}

fn test() {
    S.f();
   //~^ ERROR multiple applicable items in scope
}
