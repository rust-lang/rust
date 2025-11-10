// Suggest not mutably borrowing a mutable reference
#![crate_type = "rlib"]

pub fn f(b: &mut i32) {
    //~^ ERROR: cannot borrow
    //~| NOTE: not mutable
    //~| NOTE: the binding is already a mutable borrow
    //~| HELP: consider making the binding mutable if you need to reborrow multiple times
    h(&mut b);
    //~^ NOTE: cannot borrow as mutable
    //~| HELP: if there is only one mutable reborrow, remove the `&mut`
    g(&mut &mut b);
    //~^ NOTE: cannot borrow as mutable
    //~| HELP: if there is only one mutable reborrow, remove the `&mut`
}

pub fn g(b: &mut i32) { //~ NOTE: the binding is already a mutable borrow
    //~^ HELP: consider making the binding mutable if you need to reborrow multiple times
    h(&mut &mut b);
    //~^ ERROR: cannot borrow
    //~| NOTE: cannot borrow as mutable
    //~| HELP: if there is only one mutable reborrow, remove the `&mut`
}

pub fn h(_: &mut i32) {}

trait Foo {
    fn bar(&mut self);
}

impl Foo for &mut String {
    fn bar(&mut self) {}
}

pub fn baz(f: &mut String) { //~ HELP consider making the binding mutable
    f.bar(); //~ ERROR cannot borrow `f` as mutable, as it is not declared as mutable
    //~^ NOTE cannot borrow as mutable
}
