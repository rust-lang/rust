// check-pass
// aux-build:const_generic_issues_lib.rs
extern crate const_generic_issues_lib as lib2;
fn unused_function(
    _: <lib2::GenericType<42> as lib2::TypeFn>::Output
) {}

fn main() {}
