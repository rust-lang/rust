pub struct S(f32, f32);

pub enum E {
    V(f32, f32),
}

fn main() {
    let _x = (S { x: 1.0, y: 2.0 }, S { x: 3.0, y: 4.0 });
    //~^ ERROR struct `S` has no field named `x`
    //~| ERROR struct `S` has no field named `y`
    //~| ERROR struct `S` has no field named `x`
    //~| ERROR struct `S` has no field named `y`
    let _y = (E::V { x: 1.0, y: 2.0 }, E::V { x: 3.0, y: 4.0 });
    //~^ ERROR variant `E::V` has no field named `x`
    //~| ERROR variant `E::V` has no field named `y`
    //~| ERROR variant `E::V` has no field named `x`
    //~| ERROR variant `E::V` has no field named `y`
}
