// Test equality constraints in a where clause where the type being
// equated appears in a supertrait.

pub trait Vehicle {
    type Color;

    fn go(&self) {  }
}

pub trait Box {
    type Color;
    //
    fn mail(&self) {  }
}

pub trait BoxCar : Box + Vehicle {
}

fn dent<C:BoxCar>(c: C, color: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

fn dent_object<COLOR>(c: dyn BoxCar<Color=COLOR>) {
    //~^ ERROR ambiguous associated type
    //~| ERROR the value of the associated type `Color` (from the trait `Vehicle`) must be specified
}

fn paint<C:BoxCar>(c: C, d: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

pub fn main() { }
