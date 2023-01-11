// Variant of diverging-falllback-control-flow that tests
// the specific case of a free function with an unconstrained
// return type. This captures the pattern we saw in the wild
// in the objc crate, where changing the fallback from `!` to `()`
// resulted in unsoundness.
//
// check-pass

// revisions: nofallback fallback

#![cfg_attr(fallback, feature(never_type, never_type_fallback))]


fn make_unit() {}

trait UnitReturn {}
impl UnitReturn for i32 {}
impl UnitReturn for () {}

fn unconstrained_return<T: UnitReturn>() -> T {
    unsafe {
        let make_unit_fn: fn() = make_unit;
        let ffi: fn() -> T = std::mem::transmute(make_unit_fn);
        ffi()
    }
}

fn main() {
    // In Ye Olde Days, the `T` parameter of `unconstrained_return`
    // winds up "entangled" with the `!` type that results from
    // `panic!`, and hence falls back to `()`. This is kind of unfortunate
    // and unexpected. When we introduced the `!` type, the original
    // idea was to change that fallback to `!`, but that would have resulted
    // in this code no longer compiling (or worse, in some cases it injected
    // unsound results).
    let _ = if true { unconstrained_return() } else { panic!() };
}
