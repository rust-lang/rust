// Crate-local macro expanded `macro_export` macros cannot be accessed with module-relative paths.

macro_rules! define_exported { () => {
    #[macro_export]
    macro_rules! exported {
        () => ()
    }
}}

define_exported!();

mod m {
    use crate::exported;
    //~^ ERROR macro-expanded `macro_export` macros from the current crate cannot
    //~| WARN this was previously accepted
}

fn main() {
    crate::exported!();
    //~^ ERROR macro-expanded `macro_export` macros from the current crate cannot
    //~| WARN this was previously accepted
}
