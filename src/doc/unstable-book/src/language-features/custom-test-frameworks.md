# `custom_test_frameworks`

The tracking issue for this feature is: [#50297]

[#50297]: https://github.com/rust-lang/rust/issues/50297

------------------------

The `custom_test_frameworks` feature allows the use of `#[test_case]` and `#![test_runner]`.
Any function, const, or static can be annotated with `#[test_case]` causing it to be aggregated (like `#[test]`)
and be passed to the test runner determined by the `#![test_runner]` crate attribute.

```rust
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
```
