pub fn main() {
    let mut count = 0;
    do 999_999.times() {
        count += 1;
    }
    assert_eq!(count, 999_999);
    println!("{}", count);
}
