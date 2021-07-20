// min-lldb-version: 310

// aux-build:cross_crate_debuginfo_type_uniquing.rs
extern crate cross_crate_debuginfo_type_uniquing;

// no-prefer-dynamic
// FIXME(#86758) opt-level=1 is required to avoid triggering an
// LLVM assert on i686 during compilation
// compile-flags:-g -C lto -C opt-level=1

pub struct C;
pub fn p() -> C {
    C
}

fn main() { }
