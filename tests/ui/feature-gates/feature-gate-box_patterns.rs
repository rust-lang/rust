fn main() {
    let box x = Box::new('c'); //~ ERROR box pattern syntax is experimental
    println!("x: {}", x);
}
