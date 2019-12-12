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

// FIXME: add error code to detect this case and explain that you'll want the approach in
// `dent_object_3` of using a new type param and relying on the `where` clauses.
fn dent_object<COLOR>(c: dyn BoxCar<Color=COLOR>) {
    //~^ ERROR ambiguous associated type
    //~| ERROR the value of the associated type `Color` (from trait `Vehicle`) must be specified
}

fn paint<C:BoxCar>(c: C, d: C::Color) {
    //~^ ERROR ambiguous associated type `Color` in bounds of `C`
}

fn dent_object_2<COLOR>(c: dyn BoxCar) where <dyn BoxCar as Vehicle>::Color = COLOR {
    //~^ ERROR the value of the associated types `Color` (from trait `Vehicle`), `Color` (from
    //~| ERROR equality constraints are not yet supported in where clauses
}

fn dent_object_3<X, COLOR>(c: X)
where X: BoxCar,
    X: Vehicle<Color = COLOR>,
    X: Box<Color = COLOR>
{} // OK!

pub fn main() { }
