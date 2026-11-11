mod m {
    pub struct Foo { x: u8 }

    pub struct Bar(u8);
}

use m::{Foo, Bar};

fn main() {
    let x = Foo { x: 12 };
    let y = Bar(12);
    //~^ ERROR cannot initialize a tuple struct which contains private fields [E0423]
}
