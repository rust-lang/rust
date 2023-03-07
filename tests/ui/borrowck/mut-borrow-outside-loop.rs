// ensure borrowck messages are correct outside special case
#![feature(rustc_attrs)]
fn main() { #![rustc_error] // rust-lang/rust#49855
    let mut void = ();

    let first = &mut void;
    let second = &mut void; //~ ERROR cannot borrow
    first.use_mut();
    second.use_mut();

    loop {
        let mut inner_void = ();

        let inner_first = &mut inner_void;
        let inner_second = &mut inner_void; //~ ERROR cannot borrow
        inner_second.use_mut();
        inner_first.use_mut();
    }
}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
