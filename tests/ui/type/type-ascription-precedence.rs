// Operator precedence of type ascription
// Type ascription has very high precedence, the same as operator `as`
use std::ops::*;

struct S;
struct Z;

impl Add<Z> for S {
    type Output = S;
    fn add(self, _rhs: Z) -> S { panic!() }
}
impl Mul<Z> for S {
    type Output = S;
    fn mul(self, _rhs: Z) -> S { panic!() }
}
impl Neg for S {
    type Output = Z;
    fn neg(self) -> Z { panic!() }
}
impl Deref for S {
    type Target = Z;
    fn deref(&self) -> &Z { panic!() }
}

fn main() {
    &S: &S; //~ ERROR expected one of
    (&S): &S; // OK
    &(S: &S);

    *S: Z; // OK
    (*S): Z; // OK
    *(S: Z);

    -S: Z; // OK
    (-S): Z; // OK
    -(S: Z);

    S + Z: Z; // OK
    S + (Z: Z); // OK
    (S + Z): Z;

    S * Z: Z; // OK
    S * (Z: Z); // OK
    (S * Z): Z;

    S .. S: S; // OK
    S .. (S: S); // OK
    (S .. S): S;
}
