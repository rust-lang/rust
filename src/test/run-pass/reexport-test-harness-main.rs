// compile-flags:--test

#![reexport_test_harness_main = "test_main"]

#[cfg(test)]
fn _unused() {
    // should resolve to the entry point function the --test harness
    // creates.
    test_main();
}
