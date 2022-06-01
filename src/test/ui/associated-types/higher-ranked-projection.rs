// ignore-compare-mode-nll
// revisions: good badbase badnll
//[good] check-pass
// [badnll]compile-flags: -Zborrowck=mir

trait Mirror {
    type Image;
}

impl<T> Mirror for T {
    type Image = T;
}

#[cfg(any(badbase, badnll))]
fn foo<U, T>(_t: T)
    where for<'a> &'a T: Mirror<Image=U>
{}

#[cfg(good)]
fn foo<U, T>(_t: T)
    where for<'a> &'a T: Mirror<Image=&'a U>
{}

fn main() {
    foo(());
    //[badbase]~^ ERROR mismatched types
    //[badnll]~^^ ERROR mismatched types
}
