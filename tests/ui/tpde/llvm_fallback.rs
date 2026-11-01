//@ revisions: try only
//@ needs-tpde
//
//@[try] build-pass
//@[only] build-fail
//
//@[try] compile-flags: -Zllvm-tpde
//@[only] compile-flags: -Zllvm-tpde=only

//[only]~? ERROR: could not write output to $TEST_BUILD_DIR/llvm_fallback.llvm_fallback.bef9b9dab1dfd663-cgu.0.rcgu.o: Failed to compile module llvm_fallback.bef9b9dab1dfd663-cgu.0 with TPDE

// NOTE: if TPDE gains f16 support, this test will need to be adjusted

#![feature(f16)]

fn main() {
    if std::hint::black_box(0f16) == std::hint::black_box(7f16) {
        println!("same");
    } else {
        println!("different");
    }
}
