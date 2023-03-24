fn main() {
    for i in 0..(256 as u8) { //~ ERROR range endpoint is out of range
        println!("{}", i);
    }
}
