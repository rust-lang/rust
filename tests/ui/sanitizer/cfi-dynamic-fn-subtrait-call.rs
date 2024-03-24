// Verifies that calling a dynamic Fn subtrait object works.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Cprefer-dynamic=off -Ctarget-feature=-crt-static -Zsanitizer=cfi -Copt-level=0
//@ run-pass

trait FnSubtrait: Fn() {}
impl<T: Fn()> FnSubtrait for T {}

fn call_dynamic_fn_subtrait(f: &dyn FnSubtrait) {
     f();
}

fn main() {
    call_dynamic_fn_subtrait(&|| {});
}
