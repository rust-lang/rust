// run-pass

struct A;

impl A {
    const fn banana() -> bool {
        true
    }
}

const ABANANA: bool = A::banana();

fn main() {
    match true {
        ABANANA => {},
        _ => panic!("what?")
    }
}
