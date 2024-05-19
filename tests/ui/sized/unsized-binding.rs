fn main() {
    let x = *""; //~ ERROR E0277
    println!("{}", x);
    println!("{}", x);
}
