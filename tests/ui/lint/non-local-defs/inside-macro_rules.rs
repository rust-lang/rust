//@ check-pass
//@ edition:2021

#![warn(non_local_definitions)]

macro_rules! m {
    () => {
        trait MacroTrait {}
        struct OutsideStruct;
        fn my_func() {
            impl MacroTrait for OutsideStruct {}
            //~^ WARN non-local `impl` definition
        }
    }
}

m!();

fn main() {}
