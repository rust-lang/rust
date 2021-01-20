// check-pass
#![allow(non_upper_case_globals)]

const or: usize = 1;
const and: usize = 2;

mod or {
    pub const X: usize = 3;
}

mod and {
    pub const X: usize = 4;
}

fn main() {
    match 0 {
        0 => {}
        or => {}
        and => {}
        or::X => {}
        and::X => {}
        _ => {}
    }
}
