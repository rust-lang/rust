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

fn dent_object<COLOR>(c: &dyn BoxCar<Color=COLOR>) {
    //~^ ERROR ambiguous associated type
    //~| ERROR the value of the associated types
}

fn paint<C:BoxCar>(c: C, d: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

fn dent_object_2<COLOR>(c: &dyn BoxCar) where <dyn BoxCar as Vehicle>::Color = COLOR {
    //~^ ERROR the value of the associated types
    //~| ERROR equality constraints are not yet supported in `where` clauses
}

fn dent_object_3<X, COLOR>(c: X)
where X: BoxCar,
    X: Vehicle<Color = COLOR>,
    X: Box<Color = COLOR>
{} // OK!

pub fn main() { }
