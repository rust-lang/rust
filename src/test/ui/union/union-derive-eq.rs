#![feature(untagged_unions)]

#[derive(Eq)] // OK
union U1 {
    a: u8,
}

impl PartialEq for U1 { fn eq(&self, rhs: &Self) -> bool { true } }

#[derive(PartialEq)]
struct PartialEqNotEq;

#[derive(Eq)]
union U2 {
    a: PartialEqNotEq, //~ ERROR the trait bound `PartialEqNotEq: std::cmp::Eq` is not satisfied
}

impl PartialEq for U2 { fn eq(&self, rhs: &Self) -> bool { true } }

fn main() {}
