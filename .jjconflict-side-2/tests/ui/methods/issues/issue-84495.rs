fn main() {
    let x: i32 = 1;
    println!("{:?}", x.count()); //~ ERROR is not an iterator
}
