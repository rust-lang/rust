// run-pass
pub fn main() {
    let mut count = 0;
    for _ in 0..999_999 { count += 1; }
    assert_eq!(count, 999_999);
    println!("{}", count);
}
