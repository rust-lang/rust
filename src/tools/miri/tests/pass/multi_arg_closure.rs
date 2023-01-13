fn foo(f: &mut dyn FnMut(isize, isize) -> isize) -> isize {
    f(1, 2)
}

fn main() {
    let z = foo(&mut |x, y| x * 10 + y);
    assert_eq!(z, 12);
}
