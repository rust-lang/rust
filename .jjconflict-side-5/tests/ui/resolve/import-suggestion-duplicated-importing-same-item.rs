mod A {
    pub struct A;
    //~^ NOTE ...and refers to the unit struct `A` which is defined here
    //~| NOTE you could import this directly
}

mod B {
    use crate::A::A;
    //~^ NOTE the unit struct import `A` is defined here...
}

fn main() {
    B::A;
    //~^ ERROR unit struct import `A` is private [E0603]
    //~| NOTE private unit struct import
}
