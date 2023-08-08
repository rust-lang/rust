// run-rustfix

fn main() {
    for _ in 0..256 as u8 {} //~ ERROR range endpoint is out of range
    for _ in 0..(256 as u8) {} //~ ERROR range endpoint is out of range
}
