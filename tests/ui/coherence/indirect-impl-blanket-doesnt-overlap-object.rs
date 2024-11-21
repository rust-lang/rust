//@ check-pass

// Make sure that if we don't disqualify a built-in object impl
// due to a blanket with a trait bound that will never apply to
// the object.

pub trait SimpleService {
    type Resp;
}

trait Service {
    type Resp;
}

impl<S> Service for S where S: SimpleService + ?Sized {
    type Resp = <S as SimpleService>::Resp;
}

fn implements_service(x: &(impl Service<Resp = ()> + ?Sized)) {}

fn test(x: &dyn Service<Resp = ()>) {
    implements_service(x);
}

fn main() {}
