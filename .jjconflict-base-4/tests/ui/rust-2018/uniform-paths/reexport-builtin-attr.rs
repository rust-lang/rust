//! Regression test for <https://github.com/rust-lang/rust/issues/74236>.
//! Test renamed attr path in module doesn't ICE when printing the path.
//@ edition:2018
//@ aux-build:reexport-builtin-attr.rs
//@ compile-flags:--extern reexport_builtin_attr

fn main() {
    // Trigger an error that will print the path of dep::private::Pub (as "dep::Renamed").
    let () = reexport_builtin_attr::Renamed;
    //~^ ERROR mismatched types
}
