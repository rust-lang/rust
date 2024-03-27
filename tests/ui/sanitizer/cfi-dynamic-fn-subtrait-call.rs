// Verifies that calling a dynamic Fn subtrait object works.
//
// FIXME(#122848): Remove only-linux when fixed.
//@ only-linux
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Copt-level=0 -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi
//@ run-pass

trait FnSubtrait: Fn() {}
impl<T: Fn()> FnSubtrait for T {}

fn call_dynamic_fn_subtrait(f: &dyn FnSubtrait) {
     f();
}

fn main() {
    call_dynamic_fn_subtrait(&|| {});
}
