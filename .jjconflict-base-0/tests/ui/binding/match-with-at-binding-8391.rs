// https://github.com/rust-lang/rust/issues/8391
//@ run-pass

fn main() {
    let x = match Some(1) {
        ref _y @ Some(_) => 1,
        None => 2,
    };
    assert_eq!(x, 1);
}
