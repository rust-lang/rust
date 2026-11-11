//@ run-pass


fn two<F>(mut it: F) where F: FnMut(isize) { it(0); it(1); }

pub fn main() {
    let mut a: Vec<isize> = vec![-1, -1, -1, -1];
    let mut p: isize = 0;
    two(|i| {
        two(|j| { a[p as usize] = 10 * i + j; p += 1; })
    });
    assert_eq!(a[0], 0);
    assert_eq!(a[1], 1);
    assert_eq!(a[2], 10);
    assert_eq!(a[3], 11);
}
