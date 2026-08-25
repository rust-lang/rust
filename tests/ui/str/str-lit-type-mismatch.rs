fn main() {
    let x: &[u8] = "foo"; //~ ERROR mismatched types
    let y: &[u8; 4] = "baaa"; //~ ERROR mismatched types
    let z: &str = b"foo"; //~ ERROR mismatched types
}
