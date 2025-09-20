fn main() {
    let mut v = vec![Box::new(0u64), Box::new(1u64)];
    for item in v.extract_if(.., |x| **x == 0) {
        drop(item);
    }
}
