//@revisions: noopt opt
//@ build-fail
//@[noopt] compile-flags: -Copt-level=0
//@[opt] compile-flags: -O

#![crate_type = "lib"]
#![feature(inline_const)]

// Will be `inline` with optimizations, so then `g::<()>` becomes reachable. At the same time `g` is
// not "mentioned" in `f` since it is only called inside an inline `const` and hence never appears
// in its MIR! This fundamentally is an issue caused by reachability walking the HIR (before inline
// `const` are extracted) and collection being based on MIR.
pub fn f() {
    loop {}; const { g::<()>() }
}

// When this comes reachable (for any `T`), so does `hidden_root`.
// And `hidden_root` is monomorphic so it can become a root!
const fn g<T>() {
    match std::mem::size_of::<T>() {
        0 => (),
        _ => hidden_root(),
    }
}

#[inline(never)]
const fn hidden_root() {
    fail::<()>();
}

#[inline(never)]
const fn fail<T>() {
    // Hiding this in a generic fn so that it doesn't get evaluated by
    // MIR passes.
    const { panic!(); } //~ERROR: evaluation of `fail::<()>::{constant#0}` failed
}
