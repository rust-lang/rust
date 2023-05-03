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

fn test1() {
    &S: &S; //~ ERROR expected one of
    (&S): &S;
    &(S: &S);
}

fn test2() {
    *(S: Z); //~ ERROR expected identifier
}

fn test3() {
    -(S: Z); //~ ERROR expected identifier
}

fn test4() {
    (S + Z): Z; //~ ERROR expected one of
}

fn test5() {
    (S * Z): Z; //~ ERROR expected one of
}

fn test6() {
    S .. S: S; //~ ERROR expected identifier, found `:`
}

fn test7() {
    (S .. S): S; //~ ERROR expected one of
}

fn main() {}
