#![feature(rustc_attrs)]

#[rustc_symbol_name] //~ ERROR _ZN5basic4main
#[rustc_item_path] //~ ERROR item-path(main)
fn main() {
}
