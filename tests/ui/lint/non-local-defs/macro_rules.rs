//@ check-pass
//@ edition:2021
//@ aux-build:non_local_macro.rs
//@ rustc-env:CARGO_CRATE_NAME=non_local_def

extern crate non_local_macro;

const B: u32 = {
    #[macro_export]
    macro_rules! m0 { () => { } };
    //~^ WARN non-local `macro_rules!` definition

    1
};

non_local_macro::non_local_macro_rules!(my_macro);
//~^ WARN non-local `macro_rules!` definition

fn main() {
    #[macro_export]
    macro_rules! m { () => { } };
    //~^ WARN non-local `macro_rules!` definition

    struct InsideMain;

    impl InsideMain {
        fn bar() {
            #[macro_export]
            macro_rules! m2 { () => { } };
            //~^ WARN non-local `macro_rules!` definition
        }
    }
}
