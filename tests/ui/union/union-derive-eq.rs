#[derive(Eq)] // OK
union U1 {
    a: u8,
}

impl PartialEq for U1 { fn eq(&self, rhs: &Self) -> bool { true } }

#[derive(PartialEq, Copy, Clone)]
struct PartialEqNotEq;

#[derive(Eq)]
union U2 {
    a: PartialEqNotEq, //~ ERROR trait `Eq` is not implemented for `PartialEqNotEq`
}

impl PartialEq for U2 { fn eq(&self, rhs: &Self) -> bool { true } }

fn main() {}
