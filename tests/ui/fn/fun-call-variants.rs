//@ run-pass

fn ho<F>(f: F) -> isize where F: FnOnce(isize) -> isize { let n: isize = f(3); return n; }

fn direct(x: isize) -> isize { return x + 1; }

pub fn main() {
    let a: isize = direct(3); // direct
    let b: isize = ho(direct); // indirect unbound

    assert_eq!(a, b);
}
