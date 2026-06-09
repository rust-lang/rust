// Used to ICE; see <https://github.com/rust-lang/rust/issues/121097>

#![feature(repr_simd)]

#[repr(simd)] //~ ERROR attribute cannot be used on
enum Aligned {
    Zero = 0,
    One = 1,
}

fn tou8(al: Aligned) -> u8 {
    al as u8
}

fn main() {}
