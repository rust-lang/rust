//@ revisions: func object clause ok
//@[ok] check-pass

#![allow(dead_code)]

trait Foo<'a> {
    type Item;
}

impl<'a> Foo<'a> for() {
    type Item = ();
}

// Check that appearing in a projection input in the argument is not enough:
#[cfg(func)]
fn func1(_: for<'a> fn(<() as Foo<'a>>::Item) -> &'a i32) {
    //[func]~^ ERROR E0581
}

// Check that appearing in a projection input in the return still
// causes an error:
#[cfg(func)]
fn func2(_: for<'a> fn() -> <() as Foo<'a>>::Item) {
    //[func]~^ ERROR E0581
}

#[cfg(object)]
fn object1(_: Box<dyn for<'a> Fn(<() as Foo<'a>>::Item) -> &'a i32>) {
    //[object]~^ ERROR E0582
}

#[cfg(object)]
fn object2(_: Box<dyn for<'a> Fn() -> <() as Foo<'a>>::Item>) {
    //[object]~^ ERROR E0582
}

#[cfg(clause)]
fn clause1<T>() where T: for<'a> Fn(<() as Foo<'a>>::Item) -> &'a i32 {
    //[clause]~^ ERROR `Output` references lifetime `'a`
}

#[cfg(clause)]
fn clause2<T>() where T: for<'a> Fn() -> <() as Foo<'a>>::Item {
    //[clause]~^ ERROR `Output` references lifetime `'a`
}

fn main() { }
