mod m {
    pub struct S(crate::P);
}

use m::S;

struct P;

impl P {
    fn foo(self) {
        S(self);
        //~^ ERROR cannot initialize a tuple struct which contains private fields [E0423]
    }
}

fn main() {}
