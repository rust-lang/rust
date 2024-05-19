//@ run-pass
// Issue #12512.

//@ pretty-expanded FIXME #23616

fn main() {
    let mut foo = Vec::new();
    #[allow(unused_labels)]
    'foo: for i in &[1, 2, 3] {
        foo.push(*i);
    }
}
