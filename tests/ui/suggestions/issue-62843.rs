fn main() {
    let line = String::from("abc");
    let pattern = String::from("bc");
    println!("{:?}", line.find(pattern)); //~ ERROR E0277
}
