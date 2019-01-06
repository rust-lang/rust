#[repr(simd)] //~ ERROR are experimental
struct Coord {
    x: u32,
    y: u32,
}

fn main() {}
