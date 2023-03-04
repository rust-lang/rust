#![warn(clippy::large_stack_arrays)]
#![allow(clippy::large_enum_variant)]

#[derive(Clone, Copy)]
struct S {
    pub data: [u64; 32],
}

#[derive(Clone, Copy)]
enum E {
    S(S),
    T(u32),
}

pub static DOESNOTLINT: [u8; 512_001] = [0; 512_001];
pub static DOESNOTLINT2: [u8; 512_001] = {
    let x = 0;
    [x; 512_001]
};

fn main() {
    let bad = (
        [0u32; 20_000_000],
        [S { data: [0; 32] }; 5000],
        [Some(""); 20_000_000],
        [E::T(0); 5000],
        [0u8; usize::MAX],
    );

    let good = (
        [0u32; 1000],
        [S { data: [0; 32] }; 1000],
        [Some(""); 1000],
        [E::T(0); 1000],
        [(); 20_000_000],
    );
}
