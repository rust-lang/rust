#![warn(clippy::missing_enforced_import_renames)]

use std::alloc as colla;
use std::option::Option as Maybe;
use std::process::{exit as wrong_exit, Child as Kid};
use std::thread::sleep;
#[rustfmt::skip]
use std::{
    any::{type_name, Any},
    clone,
    sync :: Mutex,
};

fn main() {
    use std::collections::BTreeMap as OopsWrongRename;
}
