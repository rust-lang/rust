#![feature(type_alias_impl_trait, rustc_attrs)]

mod bar {
    pub type Debuggable = impl core::fmt::Debug;

    pub fn foo() -> Debuggable {
        0u32
    }
}
use bar::*;

static mut TEST: Option<Debuggable> = None;

#[rustc_error]
fn main() {
    //~^ ERROR
    unsafe { TEST = Some(foo()) }
}
