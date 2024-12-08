//@ build-pass
//@ compile-flags: -C codegen-units=2 --emit asm

fn one() -> usize {
    1
}

pub mod a {
    pub fn two() -> usize {
        ::one() + ::one()
    }
}

pub mod b {
    pub fn three() -> usize {
        ::one() + ::a::two()
    }
}

fn main() {
    a::two();
    b::three();
}
