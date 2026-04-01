// Tests whether only one EII attribute cane be applied to a signature.
#![feature(extern_item_impls)]

#[eii(a)]
#[eii(b)]
//~^ ERROR `#[eii]` can only be specified once
fn a(x: u64);

#[a]
fn implementation(x: u64) {
    println!("{x:?}")
}

// what you would write:
fn main() {
    a(42);
}
