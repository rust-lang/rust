//@ build-pass
//@ needs-rustc-debug-assertions
//@ compile-flags: -Zmir-enable-passes=+GVN -Zmir-enable-passes=+JumpThreading --crate-type=lib

pub fn fun(terminate: bool) {
    while true {}
    //~^ WARN denote infinite loops with `loop { ... }`

    while !terminate {}
}
