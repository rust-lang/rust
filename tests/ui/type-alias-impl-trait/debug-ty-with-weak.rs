//@ compile-flags: --crate-type=lib -Cdebuginfo=2
//@ build-pass

#![feature(type_alias_impl_trait)]

pub type Debuggable = impl core::fmt::Debug;
#[defines(Debuggable)]
fn foo() -> Debuggable {
    0u32
}

static mut TEST: Option<Debuggable> = None;
