// compile-flags: -Z mir-opt-level=3 -Z inline-mir
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
#![crate_type = "lib"]

// EMIT_MIR issue_78442.bar.RevealAll.diff
// EMIT_MIR issue_78442.bar.Inline.diff
pub fn bar<P>(
    // Error won't happen if "bar" is not generic
    _baz: P,
) {
    hide_foo()();
}

fn hide_foo() -> impl Fn() {
    // Error won't happen if "iterate" hasn't impl Trait or has generics
    foo
}

fn foo() { // Error won't happen if "foo" isn't used in "iterate" or has generics
}
