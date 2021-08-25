// Checks whether declaring a lang item with the wrong number
// of generic arguments crashes the compiler (issue #83893, #87573, and part of #9307).

#![feature(lang_items, no_core)]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait MySized {}

#[lang = "add"]
trait MyAdd<'a, T> {}
//~^^ ERROR: `add` language item must be applied to a trait with 1 generic argument [E0718]

#[lang = "drop_in_place"]
//~^ ERROR `drop_in_place` language item must be applied to a function with at least 1 generic
fn my_ptr_drop() {}

#[lang = "index"]
trait MyIndex<'a, T> {}
//~^^ ERROR: `index` language item must be applied to a trait with 1 generic argument [E0718]

#[lang = "phantom_data"]
//~^ ERROR `phantom_data` language item must be applied to a struct with 1 generic argument
struct MyPhantomData<T, U>;
//~^ ERROR parameter `T` is never used
//~| ERROR parameter `U` is never used

fn ice() {
    // Use add
    let r = 5;
    let a = 6;
    r + a;

    // Use drop in place
    my_ptr_drop();

    // Use index
    let arr = [0; 5];
    let _ = arr[2];

    // Use phantomdata
    let _ = MyPhantomData::<(), i32>;
}
