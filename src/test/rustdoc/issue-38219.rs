// compile-flags:--test
// should-fail

/// ```
/// fail
/// ```
#[macro_export]
macro_rules! foo { () => {} }
