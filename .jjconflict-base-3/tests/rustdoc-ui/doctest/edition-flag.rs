//@ compile-flags:--test
//@ edition:2018
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// ```rust
/// fn main() {
///     let _ = async { };
/// }
/// ```
fn main() {
    let _ = async { };
}
