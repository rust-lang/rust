pub fn main() {
    let mut count = 0;
    for _ in range(0, 999_999u) {
        count += 1;
    }
    assert_eq!(count, 999_999);
    printfln!("%u", count);
}
