//@ compile-flags: --crate-type=lib -Cdebuginfo=2
//@ build-pass

#![feature(type_alias_impl_trait)]

mod bar {
    pub type Debuggable = impl core::fmt::Debug;
    fn foo() -> Debuggable {
        0u32
    }
}
use bar::Debuggable;

static mut TEST: Option<Debuggable> = None;
