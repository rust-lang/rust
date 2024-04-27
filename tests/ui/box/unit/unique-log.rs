//@ run-pass

pub fn main() {
    let i: Box<_> = Box::new(100);
    println!("{}", i);
}
