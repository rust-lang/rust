struct S(usize, usize, usize, usize);

fn main() {
    if let S { a, b, c, d } = S(1, 2, 3, 4) {
    //~^ ERROR tuple variant `S` written as struct variant
        println!("hi");
    }
}
