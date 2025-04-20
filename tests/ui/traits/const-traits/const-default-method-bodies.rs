//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
trait ConstDefaultFn: Sized {
    (const) fn b(self);

    (const) fn a(self) {
        self.b();
    }
}

struct NonConstImpl;
struct ConstImpl;

impl ConstDefaultFn for NonConstImpl {
    fn b(self) {}
}

impl const ConstDefaultFn for ConstImpl {
    (const) fn b(self) {}
}

const fn test() {
    NonConstImpl.a();
    //~^ ERROR the trait bound `NonConstImpl: ~const ConstDefaultFn` is not satisfied
    ConstImpl.a();
}

fn main() {}
