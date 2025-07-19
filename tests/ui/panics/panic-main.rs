//@ revisions: default abort-zero abort-one abort-full unwind-zero unwind-one unwind-full

//@[abort-zero] compile-flags: -Cpanic=abort
//@[abort-zero] no-prefer-dynamic
//@[abort-zero] exec-env:RUST_BACKTRACE=0

//@[abort-one] compile-flags: -Cpanic=abort
//@[abort-one] no-prefer-dynamic
//@[abort-one] exec-env:RUST_BACKTRACE=1

//@[abort-full] compile-flags: -Cpanic=abort
//@[abort-full] no-prefer-dynamic
//@[abort-full] exec-env:RUST_BACKTRACE=full

//@[unwind-zero] compile-flags: -Cpanic=unwind
//@[unwind-zero] exec-env:RUST_BACKTRACE=0
//@[unwind-zero] needs-unwind

//@[unwind-one] compile-flags: -Cpanic=unwind
//@[unwind-one] exec-env:RUST_BACKTRACE=1
//@[unwind-one] needs-unwind

//@[unwind-full] compile-flags: -Cpanic=unwind
//@[unwind-full] exec-env:RUST_BACKTRACE=full
//@[unwind-full] needs-unwind

//@ run-fail
//@ error-pattern:moop
//@ needs-subprocess

fn main() {
    panic!("moop");
}
