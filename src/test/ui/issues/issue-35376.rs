// build-pass (FIXME(62277): could be check-pass?)
#![feature(specialization)]

fn main() {}

pub trait Alpha<T> { }

pub trait Beta {
    type Event;
}

pub trait Delta {
    type Handle;
    fn process(&self);
}

pub struct Parent<A, T>(A, T);

impl<A, T> Delta for Parent<A, T>
where A: Alpha<T::Handle>,
      T: Delta,
      T::Handle: Beta<Event = <Handle as Beta>::Event> {
    type Handle = Handle;
    default fn process(&self) {
        unimplemented!()
    }
}

impl<A, T> Delta for Parent<A, T>
where A: Alpha<T::Handle> + Alpha<Handle>,
      T: Delta,
      T::Handle: Beta<Event = <Handle as Beta>::Event> {
      fn process(&self) {
        unimplemented!()
      }
}

pub struct Handle;

impl Beta for Handle {
    type Event = ();
}
