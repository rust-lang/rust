// compile-flags: -Zmir-opt-level=2 -Zinline-mir

#![crate_type = "lib"]

// get_static is a good inlining candidate.
// But it calls a private function that is a good inlining candidate, which also accesses a private
// static.
// If we naivelly inline all of this away in MIR, we will reference a private static from this
// crate while compiling another crate, which will produce a linker error.

#[inline]
pub fn get_static() -> &'static u8 {
    get_static_impl()
}

fn get_static_impl() -> &'static u8 {
    static THING: u8 = 0;
    &THING
}

// This is the same as the get_static test, except that here we reference a local function. If we
// mishandle this we will ICE or get a linker error because the private non-exported function will
// get pulled into the other crate.

#[inline]
pub fn get_fn_ptr() -> u8 {
    wrapper()()
}

fn wrapper() -> fn() -> u8 {
    inner
}

fn inner() -> u8 {
    123
}

// Same as the get_static test, but with a promoted const. In this case, we can just generate MIR
// for all promoted consts, and permit the inlining to happen.

#[inline]
pub fn get_promoted_const() -> &'static u8 {
   inner_promoted_const()
}

fn inner_promoted_const() -> &'static u8 {
    &0
}
