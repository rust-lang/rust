struct X {
    x: isize
}

fn f1(a: &mut X, b: &mut isize, c: isize) -> isize {
    let r = a.x + *b + c;
    a.x = 0;
    *b = 10;
    return r;
}

fn f2<F>(a: isize, f: F) -> isize where F: FnOnce(isize) { f(1); return a; }

pub fn main() {
    let mut a = X {x: 1};
    let mut b = 2;
    let c = 3;
    assert_eq!(f1(&mut a, &mut b, c), 6);
    assert_eq!(a.x, 0);
    assert_eq!(b, 10);
    assert_eq!(f2(a.x, |_| a.x = 50), 0);
    assert_eq!(a.x, 50);
}
