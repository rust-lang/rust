#![feature(rustc_attrs)]

#[rustc_symbol_name]
//~^ ERROR symbol-name(_ZN5basic4main
//~| ERROR demangling(basic::main
//~| ERROR demangling-alt(basic::main)
#[rustc_def_path] //~ ERROR def-path(main)
fn main() {
}
