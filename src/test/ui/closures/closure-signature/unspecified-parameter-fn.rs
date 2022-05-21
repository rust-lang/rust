fn main() {
    let a = |a: i32, b: fn(i32) -> _| -> Vec<i32> { Vec::new() }; //~ ERROR E0282
}
