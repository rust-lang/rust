//! Ensure E0433 suggests replacing the crate's name with `crate` when the
//! current crate's name is used in a path.

mod bar {
    pub fn baz() {}
}

use current_crate_name_in_path::bar::baz;
//~^ ERROR failed to resolve: use of unresolved module or unlinked crate `current_crate_name_in_path`
//~| HELP the current crate name can not be used in a path, use the keyword `crate` instead
//~| SUGGESTION crate

fn main() {}
