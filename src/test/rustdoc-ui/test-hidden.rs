// check-pass
// compile-flags:--test
// normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
// normalize-stdout-test: "src/test/rustdoc-ui" -> "$$DIR"

/// ```
// If `const_err` becomes a hard error in the future, please replace this with another
// deny-by-default lint instead of removing it altogether
/// # ! [allow(const_err)]
/// const C: usize = 1/0;
///
/// # use std::path::PathBuf;
/// #use std::path::Path;
/// let x = Path::new("y.rs");
/// let x = PathBuf::from("y.rs");
///
/// #[cfg(FALSE)]
/// assert!(false);
///
/// # [cfg(FALSE)]
/// assert!(false);
/// ```
fn main() {
    panic!();
}
