// ignore-msvc FIXME #31306

// note that these aux-build directives must be in this order
// aux-build:svh-uta-base.rs
// aux-build:svh-utb.rs
// aux-build:svh-uta-change-use-trait.rs
// normalize-stderr-test: "(crate `(\w+)`:) .*" -> "$1 $$PATH_$2"

//! "compile-fail/svh-uta-trait.rs" is checking that we detect a
//! change from `use foo::TraitB` to use `foo::TraitB` in the hash
//! (SVH) computation (#14132), since that will affect method
//! resolution.

extern crate uta;
extern crate utb; //~ ERROR: found possibly newer version of crate `uta` which `utb` depends

fn main() {
    utb::foo()
}
