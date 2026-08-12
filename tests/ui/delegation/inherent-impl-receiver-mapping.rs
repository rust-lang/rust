#![feature(fn_delegation)]

struct X;

impl X {
    fn static_f() {}
    fn by_value(self) {}
    fn by_ref(&self) {}
    fn by_mut_ref(&mut self) {}
}

struct Y;

impl Y {
    fn get_x(&self) -> X { X }
    reuse X::{static_f, by_value, by_ref, by_mut_ref} { self.get_x() }
}

fn main() {
    let y = Y;
    y.by_ref();
    y.by_mut_ref();
    //~^ ERROR: cannot borrow `y` as mutable, as it is not declared as mutable
    y.by_value();

    let y = &Y;
    y.by_value();
    //~^ ERROR: cannot move out of `*y` which is behind a shared reference
    y.by_ref();
    y.by_mut_ref();
    //~^ ERROR: cannot borrow `*y` as mutable, as it is behind a `&` reference

    let y = &mut Y;
    y.by_value();
    //~^ ERROR: cannot move out of `*y` which is behind a mutable reference
    y.by_ref();
    y.by_mut_ref();
}
