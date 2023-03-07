// Names of public modules in libstd and libcore don't accidentally get into prelude
// because macros with the same names are in prelude.

fn main() {
    env::current_dir; //~ ERROR use of undeclared crate or module `env`
    type A = panic::PanicInfo; //~ ERROR use of undeclared crate or module `panic`
    type B = vec::Vec<u8>; //~ ERROR use of undeclared crate or module `vec`
}
