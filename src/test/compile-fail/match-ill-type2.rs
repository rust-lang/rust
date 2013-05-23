fn main() {
    match 1 {
        1 => 1, //~ ERROR mismatched types between arms
        2u => 1,
        _ => 2,
    };
}
