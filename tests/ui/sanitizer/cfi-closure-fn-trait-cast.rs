// Verifies that casting a closure to a Fn trait object works.
//
// FIXME(#122848): Remove only-linux when fixed.
//@ only-linux
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Copt-level=0 -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi
//@ run-pass

#![feature(fn_traits)]
fn main() {
   let f: &(dyn Fn()) = &(|| {}) as _;
   f.call(());
}
