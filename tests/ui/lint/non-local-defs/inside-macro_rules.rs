//@ check-pass
//@ edition:2021

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
