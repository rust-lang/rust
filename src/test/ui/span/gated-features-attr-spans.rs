#[repr(simd)] //~ ERROR are experimental
struct Weapon {
    name: String,
    damage: u32
}

fn main() {}
