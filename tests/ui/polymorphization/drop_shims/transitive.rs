// check-pass
// compile-flags:-Zpolymorphize=on

pub struct OnDrop<F: Fn()>(pub F);

impl<F: Fn()> Drop for OnDrop<F> {
    fn drop(&mut self) { }
}

fn bar<F: FnOnce()>(f: F) {
    let _ = OnDrop(|| ());
    f()
}

fn foo<R, S: FnOnce()>(
    _: R,
    _: S,
) {
    let bar = || {
        bar(|| {})
    };
    bar();
}

fn main() {
    foo(3u32, || {});
}
