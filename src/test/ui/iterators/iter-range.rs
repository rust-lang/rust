// run-pass


fn range_<F>(a: isize, b: isize, mut it: F) where F: FnMut(isize) {
    assert!((a < b));
    let mut i: isize = a;
    while i < b { it(i); i += 1; }
}

pub fn main() {
    let mut sum: isize = 0;
    range_(0, 100, |x| sum += x );
    println!("{}", sum);
}
