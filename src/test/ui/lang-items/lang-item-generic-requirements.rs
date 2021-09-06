// Checks that declaring a lang item with the wrong number
// of generic arguments errors rather than crashing (issue #83893, #87573, part of #9307, #79559).

#![feature(lang_items, no_core)]
#![no_core]

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

// When the `start` lang item is missing generics very odd things can happen, especially when
// it comes to cross-crate monomorphization
#[lang = "start"]
//~^ ERROR `start` language item must be applied to a function with 1 generic argument [E0718]
fn start(_: *const u8, _: isize, _: *const *const u8) -> isize {
    0
}

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

// use `start`
fn main() {}
