//@ revisions: bpass1 bpass2
//@ compile-flags: -Zquery-dep-graph -Zincremental-verify-ich -Csymbol-mangling-version=v0 -Ccodegen-units=1
//@ ignore-backends: gcc

#![feature(rustc_attrs)]

struct Item;

#[inline(never)]
fn generic<T>(value: u32) -> u32 {
    std::hint::black_box(value)
}

#[rustc_clean(cfg = "bpass2", loaded_from_disk = "symbol_name_and_sort_key")]
struct Marker;

fn main() {
    #[cfg(bpass1)]
    let value = 1;

    #[cfg(bpass2)]
    let value = 2;

    generic::<(Item, Item, &dyn Fn(Item))>(value);
}
