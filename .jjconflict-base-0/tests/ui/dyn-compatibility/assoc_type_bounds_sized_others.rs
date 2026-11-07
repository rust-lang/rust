//! This test checks that even if some associated types have
//! `where Self: Sized` bounds, those without still need to be
//! mentioned in trait objects.

trait Foo {
    type Bar
    where
        Self: Sized;
    type Bop;
}

fn foo(_: &dyn Foo) {}
//~^ ERROR the value of the associated type `Bop` in `Foo` must be specified

trait Bar {
    type Bop;
    type Bar
    where
        Self: Sized;
}

fn bar(_: &dyn Bar) {}
//~^ ERROR the value of the associated type `Bop` in `Bar` must be specified

fn main() {}
