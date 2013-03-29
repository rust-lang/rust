pub fn main() {
    let mut count = 0;
    for 999_999.times() {
        count += 1;
    }
    assert!(count == 999_999);
    io::println(fmt!("%u", count));
}
