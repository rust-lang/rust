// compile-flags:--test --test-args=--test-threads=1
// build-pass
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"

//! Crate documentation
//!
//! This is some perfectly normal code
//! ```
//! 10 print "hello...
//! ```
#![doc(codeblock_attr = "text")]

#![crate_name = "foo"]
#![crate_type = "lib"]

#[doc = r#"
Test code:
```rust
println!("I am evil")
```
"#]
#[doc(codeblock_attr = "text")]
fn test1() {}

#[doc = r#"
Test code:
```
println!("I am evil")
```
"#]
#[doc(codeblock_attr = "text")]
fn test2() {}

#[doc = r#"
Test code:
```
10 print "hello, I have a dangling quote
```
"#]
#[doc(codeblock_attr = "text")]
fn test3() {}

#[doc = r#"
Test code:

    10 print "hello, I have a dangling quote

"#]
#[doc(codeblock_attr = "text")]
fn test4() {}

#[doc = r#"
This is a test:
```text
panic!("Expected")
```
"#]
#[doc(codeblock_attr = "should_panic")]
#[doc(codeblock_attr = "rust")]
fn panic_test0() {}

/// This is a test:
/// ```text
/// panic!("Expected")
/// ```
#[doc(codeblock_attr = "should_panic")]
#[doc(codeblock_attr = "rust")]
fn panic_test1() {}

/// This is a test:
///
///     panic!("Expected")
///
#[doc(codeblock_attr = "should_panic")]
#[doc(codeblock_attr = "rust")]
fn panic_test2() {}

/// This is a test:
///
///     panic!("Expected")
///
#[doc(codeblock_attr ("rust", "should_panic"))]
fn panic_test3() {}
