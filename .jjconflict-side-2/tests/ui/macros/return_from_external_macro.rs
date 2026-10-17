//@ aux-crate: ret_from_ext=return_from_external_macro.rs

#![feature(super_let)]
extern crate ret_from_ext;

fn foo() -> impl Sized {
    drop(|| ret_from_ext::foo!());
    //~^ ERROR cannot return reference to temporary value

    ret_from_ext::foo!()
    //~^ ERROR temporary value dropped while borrowed
}
//~^ NOTE temporary value is freed at the end of this statement

fn main() {
    foo();
}
