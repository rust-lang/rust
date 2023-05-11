// Regression test for issue #24622. The older associated types code
// was erroneously assuming that all projections outlived the current
// fn body, causing this (invalid) code to be accepted.

pub trait Foo<'a> {
    type Bar;
}

impl<'a, T:'a> Foo<'a> for T {
    type Bar = &'a T;
}

fn denormalise<'a, T>(t: &'a T) -> <T as Foo<'a>>::Bar {
    t
}

pub fn free_and_use<T: for<'a> Foo<'a>,
                    F: for<'a> FnOnce(<T as Foo<'a>>::Bar)>(x: T, f: F) {
    let y;
    'body: loop { // lifetime annotations added for clarity
        's: loop { y = denormalise(&x); break }
        drop(x); //~ ERROR cannot move out of `x` because it is borrowed
        return f(y);
    }
}

pub fn main() {
}
