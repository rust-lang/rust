struct X;

// Test that the error correctly differentiate between the two unnamed lifetimes
impl std::ops::Add<&X> for &X {
    type Output = X;

    fn add(self, _rhs: Self) -> Self::Output {
        X
    }
}
//~^^^^ ERROR method not compatible with trait

// Test that the error correctly differentiate between named and the unnamed lifetimes
impl<'a> std::ops::Mul<&'a X> for &X {
    type Output = X;

    fn mul(self, _rhs: Self) -> Self::Output {
        X
    }
}
//~^^^^ ERROR method not compatible with trait

// Test that the error correctly differentiate between named and the unnamed lifetimes
impl<'a> std::ops::Sub<&X> for &'a X {
    type Output = X;

    fn sub(self, _rhs: Self) -> Self::Output {
        X
    }
}
//~^^^^ ERROR method not compatible with trait

// This should pass since the lifetime subtyping will pass typecheck
impl<'a, 'b> std::ops::Div<&'a X> for &'b X where 'a : 'b {
    type Output = X;

    fn div(self, _rhs: Self) -> Self::Output {
        X
    }
}

fn main() {}
