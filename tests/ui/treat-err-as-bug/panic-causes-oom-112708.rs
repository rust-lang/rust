//@ compile-flags: -Ztreat-err-as-bug
//@ dont-check-failure-status
//@ error-pattern: aborting due to `-Z treat-err-as-bug=1`
//@ dont-check-compiler-stderr
//@ rustc-env:RUST_BACKTRACE=0

fn main() {
    #[deny(while_true)]
    while true {} //~ ERROR denote infinite loops with `loop { ... }`
}
