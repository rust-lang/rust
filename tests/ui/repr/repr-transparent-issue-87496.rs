// Regression test for the ICE described in #87496.

//@ check-pass

#[repr(transparent)]
struct TransparentCustomZst(());
extern "C" {
    fn good17(p: TransparentCustomZst);
    //~^ WARNING: `extern` block uses type `TransparentCustomZst`, which is not FFI-safe
}

fn main() {}
