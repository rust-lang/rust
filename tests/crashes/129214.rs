//@ known-bug: rust-lang/rust#129214
//@ compile-flags: -Zvalidate-mir -Copt-level=3 --crate-type=lib

trait to_str {}

trait map<T> {
    fn map<U, F>(&self, f: F) -> Vec<U>
    where
        F: FnMut(&Box<usize>) -> U;
}
impl<T> map<T> for Vec<T> {
    fn map<U, F>(&self, mut f: F) -> Vec<U>
    where
        F: FnMut(&T) -> U,
    {
        let mut r = Vec::new();
        for i in self {
            r.push(f(i));
        }
        r
    }
}

fn foo<U, T: map<U>>(x: T) -> Vec<String> {
    x.map(|_e| "hi".to_string())
}

pub fn main() {
    assert_eq!(foo(vec![1]), ["hi".to_string()]);
}
