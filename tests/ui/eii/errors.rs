//@ compile-flags: --crate-type rlib
// Tests all the kinds of errors when EII attributes are used with wrong syntax.
#![feature(extern_item_impls)]
#![feature(decl_macro)]
#![feature(rustc_attrs)]
#![feature(eii_internals)]

#[eii_declaration(bar)] //~ ERROR `#[eii_declaration(...)]` is only valid on macros
fn hello() {
    #[eii_declaration(bar)] //~ ERROR `#[eii_declaration(...)]` is only valid on macros
    let x = 3 + 3;
}

#[eii_declaration] //~ ERROR `#[eii_declaration(...)]` expects a list of one or two elements
#[eii_declaration()] //~ ERROR `#[eii_declaration(...)]` expects a list of one or two elements
#[eii_declaration(bar, hello)] //~ ERROR expected this argument to be "unsafe"
#[eii_declaration(bar, "unsafe", hello)] //~ ERROR `#[eii_declaration(...)]` expects a list of one or two elements
#[eii_declaration(bar, hello, "unsafe")] //~ ERROR `#[eii_declaration(...)]` expects a list of one or two elements
#[eii_declaration = "unsafe"] //~ ERROR `#[eii_declaration(...)]` expects a list of one or two elements
#[eii_declaration(bar)]
#[rustc_builtin_macro(eii_shared_macro)]
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
