// Tests that converting a closure to a function pointer works
// The notable thing being tested here is that when the closure does not capture anything,
// the call method from its Fn trait takes a ZST representing its environment. The compiler then
// uses the assumption that the ZST is non-passed to reify this into a function pointer.
//
// This checks that the reified function pointer will have the expected alias set at its call-site.

//@ revisions: cfi kcfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ [cfi] needs-sanitizer-cfi
//@ [kcfi] needs-sanitizer-kcfi
//@ compile-flags: -C target-feature=-crt-static
//@ [cfi] compile-flags: -C codegen-units=1 -C lto -C prefer-dynamic=off -C opt-level=0
//@ [cfi] compile-flags: -Z sanitizer=cfi
//@ [kcfi] compile-flags: -Z sanitizer=kcfi
//@ run-pass

pub fn main() {
    let f: &fn() = &((|| ()) as _);
    f();
}
