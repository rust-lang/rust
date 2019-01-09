// Check that a trait with by-value self is considered object-safe.

// compile-pass
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

fn make_bar<T:Bar>(t: &T) -> &Bar {
    t // legal
}

fn make_bar_explicit<T:Bar>(t: &T) -> &Bar {
    t as &Bar // legal
}

fn make_baz<T:Baz>(t: &T) -> &Baz {
    t // legal
}

fn make_baz_explicit<T:Baz>(t: &T) -> &Baz {
    t as &Baz // legal
}

fn make_quux<T:Quux>(t: &T) -> &Quux {
    t
}

fn make_quux_explicit<T:Quux>(t: &T) -> &Quux {
    t as &Quux
}


fn main() {
}
