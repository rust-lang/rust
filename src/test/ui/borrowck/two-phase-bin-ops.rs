// run-pass
use std::ops::{AddAssign, SubAssign, MulAssign, DivAssign, RemAssign};
use std::ops::{BitAndAssign, BitOrAssign, BitXorAssign, ShlAssign, ShrAssign};

struct A(i32);

macro_rules! trivial_binop {
    ($Trait:ident, $m:ident) => {
        impl $Trait<i32> for A { fn $m(&mut self, rhs: i32) { self.0 = rhs; } }
    }
}

trivial_binop!(AddAssign, add_assign);
trivial_binop!(SubAssign, sub_assign);
trivial_binop!(MulAssign, mul_assign);
trivial_binop!(DivAssign, div_assign);
trivial_binop!(RemAssign, rem_assign);
trivial_binop!(BitAndAssign, bitand_assign);
trivial_binop!(BitOrAssign, bitor_assign);
trivial_binop!(BitXorAssign, bitxor_assign);
trivial_binop!(ShlAssign, shl_assign);
trivial_binop!(ShrAssign, shr_assign);

fn main() {
    let mut a = A(10);
    a += a.0;
    a -= a.0;
    a *= a.0;
    a /= a.0;
    a &= a.0;
    a |= a.0;
    a ^= a.0;
    a <<= a.0;
    a >>= a.0;
}
