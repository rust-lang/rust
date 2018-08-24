// Test that parameter cardinality or missing method error gets span exactly.

pub struct Foo;
impl Foo {
    fn zero(self) -> Foo { self }
    fn one(self, _: isize) -> Foo { self }
    fn two(self, _: isize, _: isize) -> Foo { self }
}

fn main() {
    let x = Foo;
    x.zero(0)   //~ ERROR this function takes 0 parameters but 1 parameter was supplied
     .one()     //~ ERROR this function takes 1 parameter but 0 parameters were supplied
     .two(0);   //~ ERROR this function takes 2 parameters but 1 parameter was supplied

    let y = Foo;
    y.zero()
     .take()    //~ ERROR no method named `take` found for type `Foo` in the current scope
     .one(0);
}
