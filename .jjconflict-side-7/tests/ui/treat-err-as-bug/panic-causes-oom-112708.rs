//@ compile-flags: -Ztreat-err-as-bug
//@ dont-check-failure-status
//@ dont-check-compiler-stderr
//@ rustc-env:RUST_BACKTRACE=0

fn main() {
    #[deny(while_true)]
    while true {} //~ ERROR denote infinite loops with `loop { ... }`
}

//~? RAW aborting due to `-Z treat-err-as-bug=1`
