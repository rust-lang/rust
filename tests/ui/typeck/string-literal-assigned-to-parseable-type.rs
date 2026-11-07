fn main() {
    // literal
    let str1: u16 = "29"; //~ ERROR mismatched types

    // function calls, variables, refs
    let str2: i32 = String::from("129"); //~ ERROR mismatched types
    let str3 = String::from("278");
    let str4: i32 = &str3; //~ ERROR mismatched types
}
