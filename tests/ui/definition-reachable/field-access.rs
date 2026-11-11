//@ aux-build:field-access-macro.rs

extern crate field_access_macro;

fn main() {
    let mut s = field_access_macro::get_struct();

    let try_field_access = field_access_macro::allow_field_access!(s); // Ok
    let try_field_access = field_access_macro::deny_field_access!(s);
    //~^ ERROR field `0` of struct `field_access_macro::n::Struct` is private
}
