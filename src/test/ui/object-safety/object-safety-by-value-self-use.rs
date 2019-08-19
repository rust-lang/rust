// Check that while a trait with by-value self is object-safe, we
// can't actually invoke it from an object (yet...?).

#![feature(rustc_attrs)]

trait Bar {
    fn bar(self);
}

trait Baz {
    fn baz(self: Self);
}

fn use_bar(t: Box<dyn Bar>) {
    t.bar() //~ ERROR cannot move a value of type dyn Bar
}

fn main() { }
