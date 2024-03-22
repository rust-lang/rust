// Tests that converting a closure to a function pointer works
// The notable thing being tested here is that when the closure does not capture anything,
// the call method from its Fn trait takes a ZST representing its environment. The compiler then
// uses the assumption that the ZST is non-passed to reify this into a function pointer.
//
// This checks that the reified function pointer will have the expected alias set at its call-site.

//@ needs-sanitizer-cfi
// FIXME(#122848) Remove only-linux once OSX CFI binaries work
//@ only-linux
//@ compile-flags: --crate-type=bin -Cprefer-dynamic=off -Clto -Zsanitizer=cfi
//@ compile-flags: -C target-feature=-crt-static -C codegen-units=1 -C opt-level=0
//@ run-pass

pub fn main() {
    let f: &fn() = &((|| ()) as _);
    f();
}
