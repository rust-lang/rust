struct S<T>(T);

impl<T> S<T> {
    fn new() -> Self {
        loop {}
    }

    fn constrain<F: Fn() -> T>(&self, _f: F) {}
}

fn main() {
    let s = S::new();
    let c = || true;
    s.constrain(c);
    let _: S<usize> = s;
    //~^ ERROR mismatched types
}
