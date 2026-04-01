//@ build-pass
//@ compile-flags: -Copt-level=0

// Regression test for #88043: LLVM crash when the RemoveZsts mir-opt pass is enabled.
// We should not see the error:
// `Basic Block in function '_ZN4main10take_until17h0067b8a660429bc9E' does not have terminator!`

fn bump() -> Option<usize> {
    unreachable!()
}

fn take_until(terminate: impl Fn() -> bool) {
    loop {
        if terminate() {
            return;
        } else {
            bump();
        }
    }
}

// CHECK-LABEL: @main
fn main() {
    take_until(|| true);
    f(None);
}

fn f(_a: Option<String>) -> Option<u32> {
    loop {
        g();
        ()
    }
}

fn g() -> Option<u32> { None }
