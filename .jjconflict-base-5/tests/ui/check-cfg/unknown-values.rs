// Check that no warning is emitted for unknown cfg value
//
//@ check-pass
//@ no-auto-check-cfg
//@ revisions: simple mixed with_values
//@ compile-flags: --check-cfg=cfg(simple,mixed,with_values)
//@ [simple]compile-flags: --check-cfg=cfg(foo,values(any()))
//@ [mixed]compile-flags: --check-cfg=cfg(foo) --check-cfg=cfg(foo,values(any()))
//@ [with_values]compile-flags:--check-cfg=cfg(foo,values(any())) --check-cfg=cfg(foo,values("aa"))

#[cfg(foo = "value")]
pub fn f() {}

#[cfg(foo)]
pub fn f() {}

fn main() {}
