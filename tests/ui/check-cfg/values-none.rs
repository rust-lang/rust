//@ check-pass
//
//@ no-auto-check-cfg
//@ revisions: explicit implicit simple concat_1 concat_2
//@ [explicit]compile-flags: --check-cfg=cfg(foo,values(none()))
//@ [implicit]compile-flags: --check-cfg=cfg(foo)
//@ [simple]  compile-flags: --check-cfg=cfg(foo,values(none(),"too"))
//@ [concat_1]compile-flags: --check-cfg=cfg(foo) --check-cfg=cfg(foo,values("too"))
//@ [concat_2]compile-flags: --check-cfg=cfg(foo,values("too")) --check-cfg=cfg(foo)

#[cfg(foo = "too")]
//[explicit]~^ WARNING unexpected `cfg` condition value
//[implicit]~^^ WARNING unexpected `cfg` condition value
fn foo_too() {}

#[cfg(foo = "bar")]
//~^ WARNING unexpected `cfg` condition value
fn foo_bar() {}

#[cfg(foo)]
fn foo() {}

fn main() {}
