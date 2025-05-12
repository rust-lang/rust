#![feature(rustc_private)]
#![allow(unused)]
#![deny(clippy::unnecessary_def_path)]

extern crate rustc_hir;

use rustc_hir::LangItem;

fn main() {
    const DEREF_TRAIT: [&str; 4] = ["core", "ops", "deref", "Deref"];
    //~^ unnecessary_def_path
    const DEREF_MUT_TRAIT: [&str; 4] = ["core", "ops", "deref", "DerefMut"];
    //~^ unnecessary_def_path
    const DEREF_TRAIT_METHOD: [&str; 5] = ["core", "ops", "deref", "Deref", "deref"];
    //~^ unnecessary_def_path

    // Don't lint, not a diagnostic or language item
    const OPS_MOD: [&str; 2] = ["core", "ops"];
}
