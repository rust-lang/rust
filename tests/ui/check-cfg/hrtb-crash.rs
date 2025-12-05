// https://github.com/rust-lang/rust/issues/139825
//@ compile-flags: --check-cfg=cfg(docsrs,test) --crate-type lib
//@ check-pass
struct A
where
    for<#[cfg(b)] c> u8:;
//~^ WARN: unexpected `cfg` condition name
