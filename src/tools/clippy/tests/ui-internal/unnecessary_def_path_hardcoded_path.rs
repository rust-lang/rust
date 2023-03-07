#![feature(rustc_private)]
#![allow(unused)]
#![warn(clippy::unnecessary_def_path)]

extern crate rustc_hir;

use rustc_hir::LangItem;

fn main() {
    const DEREF_TRAIT: [&str; 4] = ["core", "ops", "deref", "Deref"];
    const DEREF_MUT_TRAIT: [&str; 4] = ["core", "ops", "deref", "DerefMut"];
    const DEREF_TRAIT_METHOD: [&str; 5] = ["core", "ops", "deref", "Deref", "deref"];

    // Don't lint, not yet a diagnostic or language item
    const DEREF_MUT_TRAIT_METHOD: [&str; 5] = ["core", "ops", "deref", "DerefMut", "deref_mut"];
}
