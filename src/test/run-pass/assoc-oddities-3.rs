fn that_odd_parse(c: bool, n: usize) -> u32 {
    let x = 2;
    let a = [1, 2, 3, 4];
    let b = [5, 6, 7, 7];
    x + if c { a } else { b }[n]
}

fn main() {
    assert_eq!(4, that_odd_parse(true, 1));
    assert_eq!(8, that_odd_parse(false, 1));
}
