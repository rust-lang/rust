//@ run-pass

struct Sum(u32, u32);

impl PartialEq for Sum {
    fn eq(&self, other: &Self) -> bool { self.0 + self.1 == other.0 + other.1 }
}

impl Eq for Sum { }

#[derive(PartialEq, Eq)]
enum Eek {
    TheConst,
    UnusedByTheConst(Sum)
}

const THE_CONST: Eek = Eek::TheConst;

pub fn main() {
    match Eek::UnusedByTheConst(Sum(1,2)) {
        THE_CONST => { panic!(); }
        _ => {}
    }
}
