fn main() {
    match 1 {
        1..2u => 1, //~ ERROR mismatched types in range
        _ => 2,
    };
}
