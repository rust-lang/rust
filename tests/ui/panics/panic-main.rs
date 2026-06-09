//@ revisions: default abort-zero abort-one abort-full unwind-zero unwind-one unwind-full

//@[default] run-fail

//@[abort-zero] compile-flags: -Cpanic=abort
//@[abort-zero] no-prefer-dynamic
//@[abort-zero] exec-env:RUST_BACKTRACE=0
//@[abort-zero] run-crash

//@[abort-one] compile-flags: -Cpanic=abort
//@[abort-one] no-prefer-dynamic
//@[abort-one] exec-env:RUST_BACKTRACE=1
//@[abort-one] run-crash

//@[abort-full] compile-flags: -Cpanic=abort
//@[abort-full] no-prefer-dynamic
//@[abort-full] exec-env:RUST_BACKTRACE=full
//@[abort-full] run-crash

//@[unwind-zero] compile-flags: -Cpanic=unwind
//@[unwind-zero] exec-env:RUST_BACKTRACE=0
//@[unwind-zero] needs-unwind
//@[unwind-zero] run-fail

//@[unwind-one] compile-flags: -Cpanic=unwind
//@[unwind-one] exec-env:RUST_BACKTRACE=1
//@[unwind-one] needs-unwind
//@[unwind-one] run-fail

//@[unwind-full] compile-flags: -Cpanic=unwind
//@[unwind-full] exec-env:RUST_BACKTRACE=full
//@[unwind-full] needs-unwind
//@[unwind-full] run-fail

//@ error-pattern:moop
//@ needs-subprocess

fn main() {
    panic!("moop");
}
