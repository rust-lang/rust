// Ensure the output of `std::any::type_name` does not change based on `-Zverbose-internals`
// Add -Awarnings to ignore unexpected warnings on StdErr, especially from external tools.
//@ run-pass
//@ edition: 2018
//@ revisions: normal verbose
//@ [verbose]compile-flags:-Zverbose-internals --verbose -Awarnings

use std::any::type_name;

fn main() {
    assert_eq!(type_name::<[u32; 0]>(), "[u32; 0]");

    struct Wrapper<const VALUE: usize>;
    assert_eq!(type_name::<Wrapper<0>>(), "issue_94187_verbose_type_name::main::Wrapper<0>");

    assert_eq!(type_name::<dyn Fn(u32) -> u32>(), "dyn core::ops::function::Fn(u32) -> u32");
}
