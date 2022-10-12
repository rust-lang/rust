#![feature(fn_traits, unboxed_closures)]

fn test<F: for<'x> FnOnce<(&'x str,)>>(_: F) {}

struct Compose<F, G>(F, G);
impl<T, F, G> FnOnce<(T,)> for Compose<F, G>
where
    F: FnOnce<(T,)>,
    G: FnOnce<(F::Output,)>,
{
    type Output = G::Output;
    extern "rust-call" fn call_once(self, (x,): (T,)) -> G::Output {
        (self.1)((self.0)(x))
    }
}

fn bad<T>(f: fn(&'static str) -> T) {
    test(Compose(f, |_| {}));
    //~^ ERROR: implementation of `FnOnce` is not general enough
}

fn main() {}
