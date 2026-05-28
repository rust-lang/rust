//@ run-pass
#![allow(non_upper_case_globals)]

/*!
 * Try to double-check that static fns have the right size (with or
 * without dummy env ptr, as appropriate) by iterating a size-2 array.
 * If the static size differs from the runtime size, the second element
 * should be read as a null or otherwise wrong pointer and crash.
 */

fn f() { }
static bare_fns: &'static [fn()] = &[f, f];
struct S<F: FnOnce()>(F);
static mut closures: &'static mut [S<fn()>] = &mut [S(f as fn()), S(f as fn())];

pub fn main() {
    unsafe {
        for &bare_fn in bare_fns { bare_fn() }
        for closure in &mut *closures {
            let S(ref mut closure) = *closure;
            (*closure)()
        }
    }
}
