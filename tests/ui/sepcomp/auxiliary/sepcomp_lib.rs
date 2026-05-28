//@ compile-flags: -C codegen-units=3 --crate-type=rlib,dylib -g

pub mod a {
    pub fn one() -> usize {
        1
    }
}

pub mod b {
    pub fn two() -> usize {
        2
    }
}

pub mod c {
    use crate::a::one;
    use crate::b::two;
    pub fn three() -> usize {
        one() + two()
    }
}
