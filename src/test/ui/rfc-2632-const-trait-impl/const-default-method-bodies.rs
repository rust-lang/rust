#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)] // FIXME is this needed?

trait ConstDefaultFn: Sized {
    fn b(self);

    #[default_method_body_is_const]
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
    //~^ ERROR the trait bound
    //~| ERROR cannot call non-const fn
    ConstImpl.a();
}

fn main() {}
