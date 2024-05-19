use std::ops::Deref;
use std::rc::Rc;

struct Value<T>(T);

pub trait Wrap<T> {
    fn wrap() -> Self;
}

impl<R, A1, A2> Wrap<fn(A1, A2) -> R> for Value<fn(A1, A2) -> R> {
    fn wrap() -> Self {
        todo!()
    }
}

impl<F, R, A1, A2> Wrap<F> for Value<Rc<dyn Fn(A1, A2) -> R>> {
    fn wrap() -> Self {
        todo!()
    }
}

impl<F> Deref for Value<Rc<F>> {
    type Target = F;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

fn main() {
    let var_fn = Value::wrap();
    //~^ ERROR type annotations needed for `Value<Rc<_>>`

    // The combination of `Value: Wrap` obligation plus the autoderef steps
    // (caused by the `Deref` impl above) actually means that the self type
    // of the method fn below is constrained to be `Value<Rc<dyn Fn(?0, ?1) -> ?2>>`.
    // However, that's only known to us on the error path -- we still need
    // to emit an ambiguity error, though.
    let _ = var_fn.clone();
}
