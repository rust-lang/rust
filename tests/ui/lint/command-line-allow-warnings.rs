// compile-flags: -A warnings

fn main() {
    // This will be overriden by the `-A warnings` command line option.
    #[warn(non_snake_case)]
    let _OwO = 0u8;

    // But this should not.
    #[deny(non_snake_case)]
    let _UwU = 0u8;
    //~^ ERROR variable `_UwU` should have a snake case name

    bar();
    baz();
}

#[warn(warnings)]
fn bar() {
    let _OwO = 0u8;
    //~^ WARN variable `_OwO` should have a snake case name
}

#[deny(warnings)]
fn baz() {
    let _OwO = 0u8;
    //~^ ERROR variable `_OwO` should have a snake case name
}
