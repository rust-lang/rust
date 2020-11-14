// Regression test for sanitizer function instrumentation passes not
// being run when compiling with new LLVM pass manager and ThinLTO.
// Note: The issue occurred only on non-zero opt-level.
//
// needs-sanitizer-support
// needs-sanitizer-address
//
// no-prefer-dynamic
// revisions: opt0 opt1
// compile-flags: -Znew-llvm-pass-manager=yes -Zsanitizer=address -Clto=thin
//[opt0]compile-flags: -Copt-level=0
//[opt1]compile-flags: -Copt-level=1
// run-fail
// error-pattern: ERROR: AddressSanitizer: stack-use-after-scope

static mut P: *mut usize = std::ptr::null_mut();

fn main() {
    unsafe {
        {
            let mut x = 0;
            P = &mut x;
        }
        std::ptr::write_volatile(P, 123);
    }
}
