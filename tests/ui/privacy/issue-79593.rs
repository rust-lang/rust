mod foo {
    pub struct Pub { private: () }

    pub enum Enum {
        Variant { x: (), y: () },
        Other
    }

    fn correct() {
        Pub {};
        //~^ ERROR missing field `private` in initializer of `Pub`
        Enum::Variant { x: () };
        //~^ ERROR missing field `y` in initializer of `Enum`
    }
}

fn correct() {
    foo::Pub {};
    //~^ ERROR cannot construct `Pub` with struct literal syntax due to private fields
}

fn wrong() {
    foo::Enum::Variant { x: () };
    //~^ ERROR missing field `y` in initializer of `Enum`
    foo::Enum::Variant { };
    //~^ ERROR missing fields `x` and `y` in initializer of `Enum`
}

fn main() {}
