// aux-build:two_macros.rs
// compile-flags:--extern non_existent

mod n {
    extern crate two_macros;
}

mod m {
    fn check() {
        two_macros::m!(); //~ ERROR failed to resolve: use of undeclared type or module `two_macros`
    }
}

macro_rules! define_std_as_non_existent {
    () => {
        extern crate std as non_existent;
        //~^ ERROR `extern crate` items cannot shadow names passed with `--extern`
    }
}
define_std_as_non_existent!();

fn main() {}
