// run-pass
#![allow(non_upper_case_globals)]

// pretty-expanded FIXME #23616

trait TheTrait { fn dummy(&self) { } }

impl TheTrait for &'static isize { }

fn foo<'a,T>(_: &'a T) where &'a T : TheTrait { }

fn bar<T>(_: &'static T) where &'static T : TheTrait { }

fn main() {
    static x: isize = 1;
    foo(&x);
    bar(&x);
}
