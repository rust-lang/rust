// compile-flags:--test -Z unstable-options
// edition:2018

#![feature(async_await)]

/// ```rust
/// #![feature(async_await)]
/// fn main() {
///     let _ = async { };
/// }
/// ```
fn main() {
    let _ = async { };
}
