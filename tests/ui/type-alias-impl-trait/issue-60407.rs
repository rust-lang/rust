#![feature(type_alias_impl_trait, rustc_attrs)]

pub type Debuggable = impl core::fmt::Debug;

#[define_opaque(Debuggable)]
pub fn foo() -> Debuggable {
    0u32
}

static mut TEST: Option<Debuggable> = None;

#[rustc_error]
fn main() {
    //~^ ERROR
    unsafe { TEST = Some(foo()) }
}
