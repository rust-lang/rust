#![feature(rustc_attrs)]

#[rustc_symbol_name] //~ ERROR _ZN5basic4main
#[rustc_def_path] //~ ERROR def-path(main)
fn main() {
}
