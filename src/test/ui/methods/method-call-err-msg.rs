// Test that parameter cardinality or missing method error gets span exactly.

pub struct Foo;
impl Foo {
    fn zero(self) -> Foo { self }
    fn one(self, _: isize) -> Foo { self }
    fn two(self, _: isize, _: isize) -> Foo { self }
    fn three<T>(self, _: T, _: T, _: T) -> Foo { self }
}

fn main() {
    let x = Foo;
    x.zero(0)   //~ ERROR arguments to this function are incorrect
     .one()     //~ ERROR arguments to this function are incorrect
     .two(0);   //~ ERROR arguments to this function are incorrect

    let y = Foo;
    y.zero()
     .take()    //~ ERROR no method named `take` found
     .one(0);
    y.three::<usize>(); //~ ERROR arguments to this function are incorrect
}
