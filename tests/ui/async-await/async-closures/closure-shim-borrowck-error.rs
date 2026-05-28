//@ compile-flags: -Zvalidate-mir --crate-type=lib -Copt-level=3
//@ edition: 2018

fn main() {}

fn needs_fn_mut<T>(mut x: impl FnMut() -> T) {
    x();
}

fn hello(x: Ty) {
    needs_fn_mut(async || {
        //~^ ERROR cannot move out of `x`
        x.hello();
    });
}

struct Ty;
impl Ty {
    fn hello(self) {}
}
