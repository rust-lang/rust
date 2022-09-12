// run-pass
// pretty-expanded FIXME #23616

pub fn main() {
    fn f() {
    }
    let _: Box<fn()> = Box::new(f as fn());
}
