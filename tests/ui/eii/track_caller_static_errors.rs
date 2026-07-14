//@ ignore-apple
// `#[track_caller]` is only valid on functions, not on EII (foreign) statics.
#![feature(extern_item_impls)]

#[track_caller] //~ ERROR `#[track_caller]` attribute cannot be used on foreign statics
#[eii(sfoo)]
static FOO: u64 = 42;

fn main() {
    println!("{FOO}");
}
