//@ compile-flags: --crate-type=lib --test --remap-path-prefix={{src-base}}/=/the/src/ --remap-path-prefix={{src-base}}\=/the/src/
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:custom-test-runner.pp

// Example taken from the unstable book.

#![feature(custom_test_frameworks)]
#![test_runner(my_runner)]

fn my_runner(tests: &[&i32]) {
    for t in tests {
        if **t == 0 {
            println!("PASSED");
        } else {
            println!("FAILED");
        }
    }
}

#[test_case]
const WILL_PASS: i32 = 0;

#[test_case]
const WILL_FAIL: i32 = 4;
