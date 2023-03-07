// run-pass
// Tests calls to closure arguments where the closure takes 1 argument.
// This is a bit tricky due to rust-call ABI.


fn foo(f: &mut dyn FnMut(isize) -> isize) -> isize {
    f(22)
}

fn main() {
    let z = foo(&mut |x| x *100);
    assert_eq!(z, 2200);
}
