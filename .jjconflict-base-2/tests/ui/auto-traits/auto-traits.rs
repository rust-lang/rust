//@ run-pass
#![allow(unused_doc_comments)]
#![feature(auto_traits)]
#![feature(negative_impls)]

auto trait Auto {}
unsafe auto trait AutoUnsafe {}

impl !Auto for bool {}
impl !AutoUnsafe for bool {}

struct AutoBool(#[allow(dead_code)] bool);

impl Auto for AutoBool {}
unsafe impl AutoUnsafe for AutoBool {}

fn take_auto<T: Auto>(_: T) {}
fn take_auto_unsafe<T: AutoUnsafe>(_: T) {}

fn main() {
    // Parse inside functions.
    auto trait AutoInner {} //~ WARN trait `AutoInner` is never used
    unsafe auto trait AutoUnsafeInner {} //~ WARN trait `AutoUnsafeInner` is never used

    take_auto(0);
    take_auto(AutoBool(true));
    take_auto_unsafe(0);
    take_auto_unsafe(AutoBool(true));

    /// Auto traits are allowed in trait object bounds.
    let _: &(dyn Send + Auto) = &0;
}
