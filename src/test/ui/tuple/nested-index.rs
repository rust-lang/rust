// run-pass

fn main () {
    let n = (1, (2, 3)).1.1;
    assert_eq!(n, 3);
}
