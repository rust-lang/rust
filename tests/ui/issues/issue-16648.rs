// run-pass
fn main() {
    let x: (isize, &[isize]) = (2, &[1, 2]);
    assert_eq!(match x {
        (0, &[_, _]) => 0,
        (1, _) => 1,
        (2, &[_, _]) => 2,
        (2, _) => 3,
        _ => 4
    }, 2);
}
