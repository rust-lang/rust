// Checks whether declaring a lang item with the wrong number
// of generic arguments crashes the compiler (issue #83893).

#![feature(lang_items,no_core)]
#![no_core]
#![crate_type="lib"]

#[lang = "sized"]
trait MySized {}

#[lang = "index"]
trait MyIndex<'a, T> {}
//~^^ ERROR: `index` language item must be applied to a trait with 1 generic argument [E0718]

fn ice() {
    let arr = [0; 5];
    let _ = arr[2];
    //~^ ERROR: cannot index into a value of type `[{integer}; 5]` [E0608]
}
