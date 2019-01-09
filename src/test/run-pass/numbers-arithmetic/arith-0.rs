// run-pass


pub fn main() {
    let a: isize = 10;
    println!("{}", a);
    assert_eq!(a * (a - 1), 90);
}
