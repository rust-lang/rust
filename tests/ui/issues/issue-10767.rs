//@ run-pass

pub fn main() {
    fn f() {
    }
    let _: Box<fn()> = Box::new(f as fn());
}
