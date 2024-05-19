//@ run-rustfix

fn func() -> u8 {
    0
}

fn main() {
    match () {
        () => func() //~ ERROR mismatched types
    }
}
