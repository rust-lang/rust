// this has to be separate to internal-unstable.rs because these tests
// have error messages pointing deep into the internals of the
// cross-crate macros, and hence need to use error-pattern instead of
// the // ~ form.

// aux-build:internal_unstable.rs
// error-pattern:use of unstable library feature 'function'
// error-pattern:use of unstable library feature 'struct_field'
// error-pattern:use of unstable library feature 'method'
// error-pattern:use of unstable library feature 'struct2_field'

#[macro_use]
extern crate internal_unstable;

fn main() {
    call_unstable_noallow!();

    construct_unstable_noallow!(0);

    |x: internal_unstable::Foo| { call_method_noallow!(x) };

    |x: internal_unstable::Bar| { access_field_noallow!(x) };
}
