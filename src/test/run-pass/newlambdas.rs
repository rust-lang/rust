// Tests for the new |args| expr lambda syntax


fn f<F>(i: isize, f: F) -> isize where F: FnOnce(isize) -> isize { f(i) }

fn g<G>(_g: G) where G: FnOnce() { }

pub fn main() {
    assert_eq!(f(10, |a| a), 10);
    g(||());
    assert_eq!(f(10, |a| a), 10);
    g(||{});
}
