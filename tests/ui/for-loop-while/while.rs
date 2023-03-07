// run-pass


pub fn main() {
    let mut x: isize = 10;
    let mut y: isize = 0;
    while y < x { println!("{}", y); println!("hello"); y = y + 1; }
    while x > 0 {
        println!("goodbye");
        x = x - 1;
        println!("{}", x);
    }
}
