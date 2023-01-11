// run-pass

fn main() {
    let x = 1;

    #[cfg(FALSE)]
    if false {
        x = 2;
    } else if true {
        x = 3;
    } else {
        x = 4;
    }
    assert_eq!(x, 1);
}
