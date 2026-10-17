union U {
    a: u16,
    b: [u8; 3],
}

fn main() {
    _ = U { b: [()] }; //~ ERROR mismatched types
}
