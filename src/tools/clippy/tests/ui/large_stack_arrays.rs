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

fn issue_10741() {
    #[derive(Copy, Clone)]
    struct Large([u32; 100_000]);

    fn build() -> Large {
        Large([0; 100_000])
    }

    let _x = [build(); 3];
    //~^ ERROR: allocating a local array larger than 512000 bytes

    let _y = [build(), build(), build()];
    //~^ ERROR: allocating a local array larger than 512000 bytes
}

fn main() {
    let bad = (
        [0u32; 20_000_000],
        //~^ ERROR: allocating a local array larger than 512000 bytes
        [S { data: [0; 32] }; 5000],
        //~^ ERROR: allocating a local array larger than 512000 bytes
        [Some(""); 20_000_000],
        //~^ ERROR: allocating a local array larger than 512000 bytes
        [E::T(0); 5000],
        //~^ ERROR: allocating a local array larger than 512000 bytes
        [0u8; usize::MAX],
        //~^ ERROR: allocating a local array larger than 512000 bytes
    );

    let good = (
        [0u32; 1000],
        [S { data: [0; 32] }; 1000],
        [Some(""); 1000],
        [E::T(0); 1000],
        [(); 20_000_000],
    );
}
