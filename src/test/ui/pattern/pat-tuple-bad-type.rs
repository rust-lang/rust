fn main() {
    let x;

    match x {
        (..) => {} //~ ERROR type annotations needed
        _ => {}
    }

    match 0u8 {
        (..) => {} //~ ERROR mismatched types
        _ => {}
    }

    x = 10;
}
