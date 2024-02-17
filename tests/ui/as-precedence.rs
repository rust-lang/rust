//@ run-pass

#[allow(unused_parens)]
fn main() {
    assert_eq!(3 as usize * 3, 9);
    assert_eq!(3 as (usize) * 3, 9);
    assert_eq!(3 as (usize) / 3, 1);
    assert_eq!(3 as usize + 3, 6);
    assert_eq!(3 as (usize) + 3, 6);
}
