//@ compile-flags: --crate-type rlib
#![feature(eii)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[core::eii_macro_for(bar)] //~ ERROR `#[eii_macro_for(...)]` is only valid on macros
fn hello() {
    #[core::eii_macro_for(bar)] //~ ERROR `#[eii_macro_for(...)]` is only valid on macros
    let x = 3 + 3;
}

#[core::eii_macro_for] //~ ERROR `#[eii_macro_for(...)]` expects a list of one or two elements
#[core::eii_macro_for()] //~ ERROR `#[eii_macro_for(...)]` expects a list of one or two elements
#[core::eii_macro_for(bar, hello)] //~ ERROR expected this argument to be "unsafe"
#[core::eii_macro_for(bar, "unsafe", hello)] //~ ERROR `#[eii_macro_for(...)]` expects a list of one or two elements
#[core::eii_macro_for(bar, hello, "unsafe")] //~ ERROR `#[eii_macro_for(...)]` expects a list of one or two elements
#[core::eii_macro_for = "unsafe"] //~ ERROR `#[eii_macro_for(...)]` expects a list of one or two elements
#[core::eii_macro_for(bar)]
#[rustc_builtin_macro(eii_macro)]
macro foo() {}

unsafe extern "Rust" {
    safe fn bar(x: u64) -> u64;
}

#[foo] //~ ERROR `#[foo]` is only valid on functions
static X: u64 = 4;
#[foo] //~ ERROR `#[foo]` is only valid on functions
const Y: u64 = 4;
#[foo] //~ ERROR `#[foo]` is only valid on functions
macro bar() {}

#[foo()]
//~^ ERROR `#[foo]` expected no arguments or a single argument: `#[foo(default)]`
#[foo(default, bar)]
//~^ ERROR `#[foo]` expected no arguments or a single argument: `#[foo(default)]`
#[foo("default")]
//~^ ERROR `#[foo]` expected no arguments or a single argument: `#[foo(default)]`
#[foo = "default"]
//~^ ERROR `#[foo]` expected no arguments or a single argument: `#[foo(default)]`
fn other(x: u64) -> u64 {
    x
}
