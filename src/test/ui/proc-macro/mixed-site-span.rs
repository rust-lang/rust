// Proc macros using `mixed_site` spans exhibit usual properties of `macro_rules` hygiene.

// aux-build:mixed-site-span.rs

#![feature(proc_macro_hygiene)]

#[macro_use]
extern crate mixed_site_span;

struct ItemUse;

fn main() {
    'label_use: loop {
        let local_use = 1;
        proc_macro_rules!();
        //~^ ERROR use of undeclared label `'label_use`
        //~| ERROR cannot find value `local_use` in this scope
        ItemDef; // OK
        local_def; //~ ERROR cannot find value `local_def` in this scope
    }
}

macro_rules! pass_dollar_crate {
    () => (proc_macro_rules!($crate);) //~ ERROR cannot find type `ItemUse` in crate `$crate`
}
pass_dollar_crate!();
