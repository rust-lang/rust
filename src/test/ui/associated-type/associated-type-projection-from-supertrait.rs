// Test equality constraints in a where clause where the type being
// equated appears in a supertrait.

pub trait Vehicle {
    type Color;

    fn go(&self) {  }
}

pub trait Car : Vehicle {
    fn honk(&self) { }
    fn chip_paint(&self, c: Self::Color) { }
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

fn dent<C:Car>(c: C, color: C::Color) { c.chip_paint(color) }
fn a() { dent(ModelT, Black); }
fn b() { dent(ModelT, Blue); } //~ ERROR mismatched types
fn c() { dent(ModelU, Black); } //~ ERROR mismatched types
fn d() { dent(ModelU, Blue); }

///////////////////////////////////////////////////////////////////////////

fn e() { ModelT.chip_paint(Black); }
fn f() { ModelT.chip_paint(Blue); } //~ ERROR mismatched types
fn g() { ModelU.chip_paint(Black); } //~ ERROR mismatched types
fn h() { ModelU.chip_paint(Blue); }

pub fn main() { }
