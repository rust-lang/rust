fn main() {
    let x: u8 = 0;
    match x { //~ ERROR non-exhaustive patterns: `_` not covered
        0 ..= 255 => {}
    }
}
