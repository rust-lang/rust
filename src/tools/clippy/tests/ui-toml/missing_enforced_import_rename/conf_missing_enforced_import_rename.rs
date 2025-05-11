#![warn(clippy::missing_enforced_import_renames)]

use std::alloc as colla;
use std::option::Option as Maybe;
use std::process::{Child as Kid, exit as wrong_exit};
//~^ missing_enforced_import_renames
use std::thread::sleep;
//~^ missing_enforced_import_renames
#[rustfmt::skip]
use std::{
    any::{type_name, Any},
    //~^ missing_enforced_import_renames
    clone,
    //~^ missing_enforced_import_renames
    sync :: Mutex,
    //~^ missing_enforced_import_renames
};

fn main() {
    use std::collections::BTreeMap as OopsWrongRename;
    //~^ missing_enforced_import_renames
}
