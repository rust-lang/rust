// run-pass

fn main() {
    let s = "Hello";
    let first = s.bytes();
    let second = first.clone();

    assert_eq!(first.collect::<Vec<u8>>(), second.collect::<Vec<u8>>())
}
