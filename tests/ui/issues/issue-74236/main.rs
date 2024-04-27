//@ edition:2018
//@ aux-build:dep.rs
//@ compile-flags:--extern dep

fn main() {
    // Trigger an error that will print the path of dep::private::Pub (as "dep::Renamed").
    let () = dep::Renamed;
    //~^ ERROR mismatched types
}
