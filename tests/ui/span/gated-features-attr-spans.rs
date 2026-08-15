#[repr(simd)] //~ ERROR are experimental
struct Coord {
    v: [u32; 2],
}

fn main() {}
