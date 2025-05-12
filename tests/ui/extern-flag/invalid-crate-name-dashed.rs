//@ compile-flags: --extern=my-awesome-library=libawesome.rlib

// In a sense, this is a regression test for issue #113035. We no longer suggest
// `pub use my-awesome-library::*;` (sic!) as we outright ban this crate name.

pub use my_awesome_library::*;

fn main() {}

//~? ERROR crate name `my-awesome-library` passed to `--extern` is not a valid ASCII identifier
//~? HELP consider replacing the dashes with underscores: `my_awesome_library`
