// Test equality constraints in a where clause where the type being
// equated appears in a supertrait.

pub trait Vehicle {
    type Color;

    fn go(&self) {  }
}

pub trait Car : Vehicle {
    fn honk(&self) { }
}

///////////////////////////////////////////////////////////////////////////

struct Black;
struct ModelT;
impl Vehicle for ModelT { type Color = Black; }
impl Car for ModelT { }

///////////////////////////////////////////////////////////////////////////

struct Blue;
struct ModelU;
impl Vehicle for ModelU { type Color = Blue; }
impl Car for ModelU { }

///////////////////////////////////////////////////////////////////////////

fn black_car<C:Car<Color=Black>>(c: C) {
}

fn blue_car<C:Car<Color=Blue>>(c: C) {
}

fn a() { black_car(ModelT); }
fn b() { blue_car(ModelT); } //~ ERROR type mismatch
fn c() { black_car(ModelU); } //~ ERROR type mismatch
fn d() { blue_car(ModelU); }

pub fn main() { }
