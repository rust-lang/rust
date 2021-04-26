//! Crate documentation
//!
#![doc(codeblock_attr = "text")]

// build-pass

#![crate_type = "lib"]

#[doc = r#"
Test code:
```rust
println!("I am evil")
```
"#]
#[doc(codeblock_attr = "text")]
fn test1() {}

/// This is a test:
/// ```text
/// panic!("Expected")
/// ```
#[doc(codeblock_attr = "should_panic")]
#[doc(codeblock_attr = "rust")]
fn panic_test() {}

/// This is a test:
///
///     panic!("Expected")
///
#[doc(codeblock_attr ("rust", "should_panic"))]
fn panic_test3() {}
