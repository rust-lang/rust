#[repr(u128)]
enum A { //~ ERROR repr with 128-bit type is unstable
    A(u64)
}

fn main() {}
