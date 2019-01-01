// run-pass

#[derive(Eq, PartialEq)]
#[repr(i8)]
pub enum A {
    B = -1,
    C = 1,
}

pub const D: A = A::B;

fn main() {
    match A::C {
        D => {},
        _ => {}
    }
}
