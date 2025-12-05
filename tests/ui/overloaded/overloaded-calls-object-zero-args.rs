//@ run-pass
// Tests calls to closure arguments where the closure takes 0 arguments.
// This is a bit tricky due to rust-call ABI.


fn foo(f: &mut dyn FnMut() -> isize) -> isize {
    f()
}

fn main() {
    let z = foo(&mut || 22);
    assert_eq!(z, 22);
}
