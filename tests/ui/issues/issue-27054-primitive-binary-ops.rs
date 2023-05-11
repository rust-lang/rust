// run-pass
fn main() {
    let x = &mut 1;
    assert_eq!(*x + { *x=2; 1 }, 2);
}
