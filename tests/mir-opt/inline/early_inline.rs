// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ test-mir-pass: ForceInline
//@ compile-flags: --crate-type=lib -C opt-level=0 -Z mir-opt-level=1

#![feature(rustc_attrs)]

pub struct MyThing(pub u32);

impl Copy for MyThing {}
impl Clone for MyThing {
    #[rustc_early_inline]
    fn clone(&self) -> Self {
        *self
    }
}

// EMIT_MIR early_inline.do_stuff.ForceInline.diff
pub fn do_stuff(mine: &MyThing) -> MyThing {
    // CHECK-LABEL: fn do_stuff(_1: &MyThing)
    // CHECK: (inlined <MyThing as Clone>::clone)
    // CHECK: _0 = copy (*_1);
    mine.clone()
}
