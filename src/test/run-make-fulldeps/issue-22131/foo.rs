/// ```rust
/// assert_eq!(foo::foo(), 1);
/// ```
#[cfg(feature = "bar")]
pub fn foo() -> i32 { 1 }
