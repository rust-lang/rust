//@ run-pass
fn main() {
    assert_eq!((||||42)()(), 42);
}
