// check-pass

#![feature(existential_type)]

existential type Debuggable: core::fmt::Debug;

static mut TEST: Option<Debuggable> = None;

fn main() {
    unsafe { TEST = Some(foo()) }
}

fn foo() -> Debuggable {
    0u32
}
