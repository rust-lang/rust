//! Regression test for https://github.com/rust-lang/rust/issues/41498
//@ run-pass
struct S;
impl S {
    fn mutate(&mut self) {}
}

fn call_and_ref<T, F: FnOnce() -> T>(x: &mut Option<T>, f: F) -> &mut T {
    *x = Some(f());
    x.as_mut().unwrap()
}

fn main() {
    let mut n = None;
    call_and_ref(&mut n, || [S])[0].mutate();
}
