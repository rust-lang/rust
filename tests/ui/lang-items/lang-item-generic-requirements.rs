// Checks that declaring a lang item with the wrong number of generic arguments errors rather than
// crashing (issue #83474, #83893, #87573, part of #9307, #79559).

#![feature(lang_items, no_core)]
#![no_core]

#[lang = "sized"]
trait MySized {}

#[lang = "add"]
trait MyAdd<'a, T> {}
//~^^ ERROR: `add` lang item must be applied to a trait with 1 generic argument [E0718]

#[lang = "drop_in_place"]
//~^ ERROR `drop_in_place` lang item must be applied to a function with at least 1 generic
fn my_ptr_drop() {}

#[lang = "index"]
trait MyIndex<'a, T> {}
//~^^ ERROR: `index` lang item must be applied to a trait with 1 generic argument [E0718]

#[lang = "phantom_data"]
//~^ ERROR `phantom_data` lang item must be applied to a struct with 1 generic argument
struct MyPhantomData<T, U>;
//~^ ERROR `T` is never used
//~| ERROR `U` is never used

#[lang = "owned_box"]
//~^ ERROR `owned_box` lang item must be applied to a struct with at least 1 generic argument
struct Foo;

// When the `start` lang item is missing generics very odd things can happen, especially when
// it comes to cross-crate monomorphization
#[lang = "start"]
//~^ ERROR `start` lang item must be applied to a function with 1 generic argument [E0718]
fn start(_: *const u8, _: isize, _: *const *const u8) -> isize {
    0
}

fn ice() {
    // Use add
    let r = 5;
    let a = 6;
    r + a; //~ ERROR cannot add

    // Use drop in place
    my_ptr_drop();

    // Use index
    let arr = [0; 5];
    let _ = arr[2];
    //~^ ERROR cannot index into a value of type `[{integer}; 5]`

    // Use phantomdata
    let _ = MyPhantomData::<(), i32>;

    // Use Foo
    let _: () = Foo;
    //~^ ERROR mismatched types
}

// use `start`
fn main() {}

//~? ERROR requires `copy` lang_item
