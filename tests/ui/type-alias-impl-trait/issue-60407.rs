#![feature(type_alias_impl_trait, rustc_attrs)]

type Debuggable = impl core::fmt::Debug;

static mut TEST: Option<Debuggable> = None;

#[rustc_error]
fn main() {
    //~^ ERROR
    unsafe { TEST = Some(foo()) }
}

fn foo() -> Debuggable {
    0u32
}
