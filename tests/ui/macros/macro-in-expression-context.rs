// run-rustfix

macro_rules! foo {
    () => {
        assert_eq!("A", "A");
        //~^ WARN trailing semicolon in macro
        //~| WARN this was previously
        //~| NOTE macro invocations at the end of a block
        //~| NOTE to ignore the value produced by the macro
        //~| NOTE for more information
        //~| NOTE `#[warn(semicolon_in_expressions_from_macros)]` on by default
        assert_eq!("B", "B");
    }
    //~^^ ERROR macro expansion ignores token `assert_eq` and any following
    //~| NOTE the usage of `foo!` is likely invalid in expression context
}

fn main() {
    foo!()
    //~^ NOTE caused by the macro expansion here
    //~| NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
    //~| NOTE in this expansion
}
