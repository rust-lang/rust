// this has to be separate to internal-unstable.rs because these tests
// have error messages pointing deep into the internals of the
// cross-crate macros, and hence need to use error-pattern instead of
// the // ~ form.

//@ aux-build:internal_unstable.rs

#[macro_use]
extern crate internal_unstable;

fn main() {
    call_unstable_noallow!(); //~ ERROR use of unstable library feature `function`

    construct_unstable_noallow!(0); //~ ERROR use of unstable library feature `struct_field`

    |x: internal_unstable::Foo| { call_method_noallow!(x) };
    //~^ ERROR use of unstable library feature `method`

    |x: internal_unstable::Bar| { access_field_noallow!(x) };
    //~^ ERROR use of unstable library feature `struct2_field`
}
