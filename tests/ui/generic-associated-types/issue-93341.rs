//@ check-pass

use std::marker::PhantomData;

pub struct Id<'id>(PhantomData<fn(&'id ()) -> &'id ()>);

fn new_id() -> Id<'static> {
    Id(PhantomData)
}

pub trait HasLifetime where {
    type AtLifetime<'a>;
}

pub struct ExistentialLifetime<S: HasLifetime>(S::AtLifetime<'static>);

impl<S: HasLifetime> ExistentialLifetime<S> {
    pub fn new<F>(f: F) -> ExistentialLifetime<S>
        where for<'id> F: FnOnce(Id<'id>) -> S::AtLifetime<'id> {
        ExistentialLifetime(f(new_id()))
    }
}


struct ExampleS<'id>(Id<'id>);

struct ExampleMarker;

impl HasLifetime for ExampleMarker {
    type AtLifetime<'id> = ExampleS<'id>;
}


fn broken0() -> ExistentialLifetime<ExampleMarker> {
    fn new_helper<'id>(id: Id<'id>) -> ExampleS<'id> {
        ExampleS(id)
    }

    ExistentialLifetime::<ExampleMarker>::new(new_helper)
}

fn broken1() -> ExistentialLifetime<ExampleMarker> {
    fn new_helper<'id>(id: Id<'id>) -> <ExampleMarker as HasLifetime>::AtLifetime<'id> {
        ExampleS(id)
    }

    ExistentialLifetime::<ExampleMarker>::new(new_helper)
}

fn broken2() -> ExistentialLifetime<ExampleMarker> {
    ExistentialLifetime::<ExampleMarker>::new(|id| ExampleS(id))
}

fn main() {}
