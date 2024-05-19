#![warn(clippy::integer_division_remainder_used)]
#![allow(unused_variables)]
#![allow(clippy::op_ref)]

struct CustomOps(pub i32);
impl std::ops::Div for CustomOps {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}
impl std::ops::Rem for CustomOps {
    type Output = Self;

    fn rem(self, rhs: Self) -> Self::Output {
        Self(self.0 % rhs.0)
    }
}

fn main() {
    // should trigger
    let a = 10;
    let b = 5;
    let c = a / b;
    let d = a % b;
    let e = &a / b;
    let f = a % &b;
    let g = &a / &b;
    let h = &10 % b;
    let i = a / &4;

    // should not trigger on custom Div and Rem
    let w = CustomOps(3);
    let x = CustomOps(4);
    let y = w / x;

    let w = CustomOps(3);
    let x = CustomOps(4);
    let z = w % x;
}
