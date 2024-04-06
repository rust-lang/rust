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

#[inline(never)]
pub fn main() {
    a::two();
    b::three();
}
