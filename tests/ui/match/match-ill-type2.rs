fn main() {
    match 1i32 {
        1i32 => 1,
        2u32 => 1, //~ ERROR mismatched types
        _ => 2,
    };
}
