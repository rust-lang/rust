//@ ignore-target: windows # does not ignore ZST arguments
//@ ignore-target: powerpc # does not ignore ZST arguments
//@ ignore-target: s390x # does not ignore ZST arguments
//@ ignore-target: sparc # does not ignore ZST arguments

// Some platforms ignore ZSTs, meaning that the argument is not passed, even though it is part
// of the callee's ABI. Test that this doesn't trip any asserts.
//
// NOTE: this  test only succeeds when the `()` argument uses `Passmode::Ignore`. For some targets,
// notably msvc, such arguments are not ignored, which would cause UB when attempting to read the
// second `i32` argument while the next item in the variable argument list is `()`.

fn main() {
    unsafe extern "C" fn variadic(mut ap: ...) {
        ap.next_arg::<i32>();
        ap.next_arg::<i32>();
    }

    unsafe { variadic(0i32, (), 1i32) }
}
