//@ run-pass

fn test<F>(f: F) -> usize
where
    F: FnOnce(usize) -> usize,
{
    return f(22);
}

pub fn main() {
    let y = test(|x| 4 * x);
    assert_eq!(y, 88);
}
