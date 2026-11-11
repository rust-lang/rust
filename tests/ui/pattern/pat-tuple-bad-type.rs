fn main() {
    let x; //~ ERROR type annotations needed

    match x {
        (..) => {}
        _ => {}
    }

    match 0u8 {
        (..) => {} //~ ERROR mismatched types
        _ => {}
    }

    x = 10;
}
