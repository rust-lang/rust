// check-pass

trait Trait {
    fn static_call(&self) where Self: Sized;
    fn maybe_dynamic_call(&self) {
        unimplemented!("unsupported maybe_dynamic_call");
    }
}

impl<T: ?Sized + Trait> Trait for &T {
    fn static_call(&self) where Self: Sized {
        (**self).maybe_dynamic_call();
    }
}

fn foo(x: &dyn Trait) {
    x.static_call();
}

fn main() {}
