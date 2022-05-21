fn main() {
    let a = |a: _, b: &i32| -> Vec<i32> { Vec::new() }; //~ ERROR E0282
}
