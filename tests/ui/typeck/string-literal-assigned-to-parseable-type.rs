fn main() {
    // literal
    let str1: u16 = "29"; //~ ERROR mismatched types

    // function calls
    let str2: i32 = String::from("129"); //~ ERROR mismatched types
}