fn main() {
    let x: u64 = 1;
    let y: u64 = x as u64; //~ WARN: unnecessary type cast
    println!("{}", y);
}
