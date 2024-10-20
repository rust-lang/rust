//@ known-bug: #131668

#![feature(generic_associated_types_extended)]
trait B {
    type Y<const N: i16>;
}

struct Erase<T: B>(T);

fn make_static() {
    Erase::<dyn for<'c> B<&'c ()>>(());
}
