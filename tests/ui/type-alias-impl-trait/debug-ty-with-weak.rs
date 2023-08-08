// compile-flags: --crate-type=lib -Cdebuginfo=2
// build-pass

#![feature(type_alias_impl_trait)]

type Debuggable = impl core::fmt::Debug;

static mut TEST: Option<Debuggable> = None;

fn foo() -> Debuggable {
    0u32
}
