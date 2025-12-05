// Names of public modules in libstd and libcore don't accidentally get into prelude
// because macros with the same names are in prelude.

fn main() {
    env::current_dir; //~ ERROR use of unresolved module or unlinked crate `env`
    type A = panic::PanicInfo; //~ ERROR use of unresolved module or unlinked crate `panic`
    type B = vec::Vec<u8>; //~ ERROR use of unresolved module or unlinked crate `vec`
}
