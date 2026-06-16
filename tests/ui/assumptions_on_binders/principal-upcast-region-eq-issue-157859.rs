//@compile-flags: -Zassumptions-on-binders  -Znext-solver=globally
//@ dont-require-annotations: ERROR

trait Super<U> {
    fn a(&self) {
        let a: &dyn Sub = &();
        let b: &dyn Super<for<'a> fn(&'a ())> = a;
    }
}

impl<T> Super<T> for () {}

trait Sub: Super<fn(&'static ())> {}

impl Sub for () {}

fn main() {
    let a: &dyn Sub = &();
}
