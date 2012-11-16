struct Bike {
    name: ~str,
}

trait BikeMethods {
    fn woops(&const self) -> ~str;
}

pub impl Bike : BikeMethods {
    static fn woops(&const self) -> ~str { ~"foo" }
    //~^ ERROR method `woops` is declared as static in its impl, but not in its trait
}

pub fn main() {
}
