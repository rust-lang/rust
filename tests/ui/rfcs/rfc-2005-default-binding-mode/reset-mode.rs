// run-pass
// Test that we "reset" the mode as we pass through a `&` pattern.
//
// cc #46688

fn surprise(x: i32) {
    assert_eq!(x, 2);
}

fn main() {
    let x = &(1, &2);
    let (_, &b) = x;
    surprise(b);
}
