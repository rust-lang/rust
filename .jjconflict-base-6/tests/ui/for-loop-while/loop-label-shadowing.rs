//@ run-pass
// Issue #12512.


fn main() {
    let mut foo = Vec::new();
    #[allow(unused_labels)]
    'foo: for i in &[1, 2, 3] {
        foo.push(*i);
    }
}
