// Checks whether declaring a lang item with the wrong number
// of generic arguments crashes the compiler (issue #83893).

#![feature(lang_items,no_core)]
#![no_core]
#![crate_type="lib"]

#[lang = "sized"]
trait MySized {}

#[lang = "add"]
trait MyAdd<'a, T> {}
//~^^ ERROR: `add` language item must be applied to a trait with 1 generic argument [E0718]

fn ice() {
    let r = 5;
    let a = 6;
    r + a
    //~^ ERROR: cannot add `{integer}` to `{integer}` [E0369]
}
