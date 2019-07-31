// check-pass

#![feature(type_alias_impl_trait)]

type Debuggable = impl core::fmt::Debug;

static mut TEST: Option<Debuggable> = None;

fn main() {
    unsafe { TEST = Some(foo()) }
}

fn foo() -> Debuggable {
    0u32
}
