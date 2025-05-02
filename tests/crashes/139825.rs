//@ known-bug: #139825
//@compile-flags: --check-cfg=cfg(docsrs,test) --crate-type lib
struct a
where
    for<#[cfg(b)] c> u8:;
