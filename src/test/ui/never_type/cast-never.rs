// Test that we can explicitly cast ! to another type

// check-pass

fn main() {
    let x: ! = panic!();
    let y: u32 = x as u32;
}
