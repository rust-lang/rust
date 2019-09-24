// Check that a trait with by-value self is considered object-safe.

// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
#![allow(trivial_casts)]

trait Bar {
    fn bar(self);
}

trait Baz {
    fn baz(self: Self);
}

trait Quux {
    // Legal because of the where clause:
    fn baz(self: Self) where Self : Sized;
}

fn make_bar<T:Bar>(t: &T) -> &dyn Bar {
    t // legal
}

fn make_bar_explicit<T:Bar>(t: &T) -> &dyn Bar {
    t as &dyn Bar // legal
}

fn make_baz<T:Baz>(t: &T) -> &dyn Baz {
    t // legal
}

fn make_baz_explicit<T:Baz>(t: &T) -> &dyn Baz {
    t as &dyn Baz // legal
}

fn make_quux<T:Quux>(t: &T) -> &dyn Quux {
    t
}

fn make_quux_explicit<T:Quux>(t: &T) -> &dyn Quux {
    t as &dyn Quux
}


fn main() {
}
