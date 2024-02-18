//@ run-pass
// Test that regionck creates the right region links in the pattern
// binding of a for loop

fn foo<'a>(v: &'a [usize]) -> &'a usize {
    for &ref x in v { return x; }
    unreachable!()
}

fn main() {
    assert_eq!(foo(&[0]), &0);
}
