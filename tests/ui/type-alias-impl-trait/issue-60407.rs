//@ check-pass

#![feature(type_alias_impl_trait)]

pub type Debuggable = impl core::fmt::Debug;

#[define_opaque(Debuggable)]
pub fn foo() -> Debuggable {
    0u32
}

static mut TEST: Option<Debuggable> = None;

fn main() {
    unsafe { TEST = Some(foo()) }
}
