// Operator precedence of type ascription
// Type ascription has very high precedence, the same as operator `as`

#![feature(type_ascription)]

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
    &S: &S; // OK
    (&S): &S; // OK
    &(S: &S); //~ ERROR mismatched types

    *S: Z; // OK
    (*S): Z; // OK
    *(S: Z); //~ ERROR mismatched types
    //~^ ERROR type `Z` cannot be dereferenced

    -S: Z; // OK
    (-S): Z; // OK
    -(S: Z); //~ ERROR mismatched types
    //~^ ERROR cannot apply unary operator `-` to type `Z`

    S + Z: Z; // OK
    S + (Z: Z); // OK
    (S + Z): Z; //~ ERROR mismatched types

    S * Z: Z; // OK
    S * (Z: Z); // OK
    (S * Z): Z; //~ ERROR mismatched types

    S .. S: S; // OK
    S .. (S: S); // OK
    (S .. S): S; //~ ERROR mismatched types
}
