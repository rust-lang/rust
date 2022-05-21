fn main() {
    let a = |a: i32, b: Vec<_>| -> Vec<i32> { Vec::new() }; //~ ERROR E0282
}
