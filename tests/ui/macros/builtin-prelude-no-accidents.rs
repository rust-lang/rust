// Names of public modules in libstd and libcore don't accidentally get into prelude
// because macros with the same names are in prelude.

fn main() {
    env::current_dir; //~ ERROR cannot find item `env`
    type A = panic::PanicInfo; //~ ERROR cannot find item `panic`
    type B = vec::Vec<u8>; //~ ERROR cannot find item `vec`
}
