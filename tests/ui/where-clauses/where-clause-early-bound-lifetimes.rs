//@ run-pass
#![allow(non_upper_case_globals)]


trait TheTrait { fn dummy(&self) { } } //~ WARN method `dummy` is never used

impl TheTrait for &'static isize { }

fn foo<'a,T>(_: &'a T) where &'a T : TheTrait { }

fn bar<T>(_: &'static T) where &'static T : TheTrait { }

fn main() {
    static x: isize = 1;
    foo(&x);
    bar(&x);
}
