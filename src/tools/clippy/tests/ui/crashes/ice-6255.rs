// originally from rustc ./tests/ui/macros/issue-78325-inconsistent-resolution.rs
// inconsistent resolution for a macro

macro_rules! define_other_core {
    ( ) => {
        extern crate std as core;
    };
}

fn main() {
    core::panic!();
}

define_other_core!();
//~^ ERROR: macro-expanded `extern crate` items cannot shadow names passed with `--extern`
