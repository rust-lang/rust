// `macro_rules` items produced by transparent macros have correct hygiene in basic cases.
// Local variables and labels are hygienic, items are not hygienic.
// `$crate` refers to the crate that defines `macro_rules` and not the outer transparent macro.

//@ proc-macro: gen-macro-rules-hygiene.rs
//@ ignore-backends: gcc

#[macro_use]
extern crate gen_macro_rules_hygiene;

struct ItemUse;

gen_macro_rules!();
//~^ ERROR use of undeclared label `'label_use`
//~| ERROR cannot find value `local_use` in this scope

fn main() {
    'label_use: loop {
        let local_use = 1;
        generated!();
        ItemDef; // OK
        local_def; //~ ERROR cannot find value `local_def` in this scope
    }
}
