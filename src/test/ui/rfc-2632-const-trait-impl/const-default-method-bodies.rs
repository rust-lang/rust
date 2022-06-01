#![feature(const_trait_impl)]

#[const_trait]
trait ConstDefaultFn: Sized {
    fn b(self);

    fn a(self) {
        self.b();
    }
}

struct NonConstImpl;
struct ConstImpl;

impl ConstDefaultFn for NonConstImpl {
    fn b(self) {}
}

impl const ConstDefaultFn for ConstImpl {
    fn b(self) {}
}

const fn test() {
    NonConstImpl.a();
    //~^ ERROR no method named `a` found for struct `NonConstImpl`
    ConstImpl.a();
}

fn main() {}
