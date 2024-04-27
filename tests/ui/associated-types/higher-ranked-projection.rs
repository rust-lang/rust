//@ revisions: good bad
//@[good] check-pass

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

fn main() {
    foo(());
    //[bad]~^ ERROR mismatched types
}
