// Verifies that casting a closure to a Fn trait object works.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0
//@ run-pass

#![feature(fn_traits)]
fn main() {
   let f: &(dyn Fn()) = &(|| {}) as _;
   f.call(());
}
