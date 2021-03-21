// check-pass

trait Trait {
    fn static_call(&mut self) where Self: Sized;
    fn maybe_dynamic_call(&self) {
        unimplemented!("unsupported maybe_dynamic_call");
    }
}

impl<T: ?Sized + Trait> Trait for &mut T {
    fn static_call(&mut self) where Self: Sized {
        (**self).maybe_dynamic_call();
    }
}

fn foo(mut x: &mut dyn Trait) {
    x.static_call();
}

fn main() {}
