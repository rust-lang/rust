//@ build-pass
//@ compile-flags: -C codegen-units=2 --emit asm

fn one() -> usize {
    1
}

pub mod a {
    pub fn two() -> usize {
        crate::one() + crate::one()
    }
}

pub mod b {
    pub fn three() -> usize {
        crate::one() + crate::a::two()
    }
}

fn main() {
    a::two();
    b::three();
}
