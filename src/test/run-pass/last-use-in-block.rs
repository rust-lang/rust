#![allow(dead_code)]
#![allow(unused_parens)]
// Issue #1818


fn lp<T, F>(s: String, mut f: F) -> T where F: FnMut(String) -> T {
    while false {
        let r = f(s);
        return (r);
    }
    panic!();
}

fn apply<T, F>(s: String, mut f: F) -> T where F: FnMut(String) -> T {
    fn g<T, F>(s: String, mut f: F) -> T where F: FnMut(String) -> T {f(s)}
    g(s, |v| { let r = f(v); r })
}

pub fn main() {}
