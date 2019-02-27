#![feature(rustc_attrs)]

// revisions: good bad

trait Mirror {
    type Image;
}

impl<T> Mirror for T {
    type Image = T;
}

#[cfg(bad)]
fn foo<U, T>(_t: T)
    where for<'a> &'a T: Mirror<Image=U>
{}

#[cfg(good)]
fn foo<U, T>(_t: T)
    where for<'a> &'a T: Mirror<Image=&'a U>
{}

#[rustc_error]
fn main() { //[good]~ ERROR compilation successful
    foo(());
    //[bad]~^ ERROR type mismatch
}
