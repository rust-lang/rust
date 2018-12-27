pub fn main() {
    let i: isize = if false { panic!() } else { 5 };
    println!("{}", i);
}
