struct Length(u8);
struct Meter;

impl core::ops::Mul<Meter> for u8 {
    type Output = Length;
    fn mul(self, _: Meter) -> Length {
        Length(self)
    }
}

#[derive(Clone, Default, PartialEq, Eq, Hash)]
struct Vec1 {
    x: i32,
}

impl core::ops::Mul<Vec1> for i32 {
    type Output = Vec1;
    fn mul(self, mut right: Vec1) -> Vec1 {
        right.x *= self;
        right
    }
}

impl core::ops::Mul<i32> for Vec1 {
    type Output = Vec1;
    fn mul(mut self, right: i32) -> Vec1 {
        self.x *= right;
        self
    }
}

#[allow(clippy::no_effect)]
#[warn(clippy::erasing_op)]
fn test(x: u8) {
    x * 0;
    //~^ ERROR: this operation will always return zero. This is likely not the intended ou
    //~| NOTE: `-D clippy::erasing-op` implied by `-D warnings`
    0 & x;
    //~^ ERROR: this operation will always return zero. This is likely not the intended ou
    0 / x;
    //~^ ERROR: this operation will always return zero. This is likely not the intended ou
    0 * Meter; // no error: Output type is different from the non-zero argument
    0 * Vec1 { x: 5 };
    //~^ ERROR: this operation will always return zero. This is likely not the intended ou
    Vec1 { x: 5 } * 0;
    //~^ ERROR: this operation will always return zero. This is likely not the intended ou
}

fn main() {
    test(0)
}
