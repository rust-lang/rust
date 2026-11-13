// originally from rustc ./tests/ui/macros/issue-78325-inconsistent-resolution.rs
// inconsistent resolution for a macro

macro_rules! define_other_core {
    ( ) => {
        extern crate std as core;
        //~^ ERROR: macro-expanded `extern crate` items cannot shadow names passed with `--extern`
    };
}

fn main() {
    core::panic!(); //~ ERROR: `core` is ambiguous
}

define_other_core!();
