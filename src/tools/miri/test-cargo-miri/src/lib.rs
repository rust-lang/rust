/// Doc-test test
///
/// ```rust
/// assert!(cargo_miri_test::make_true());
/// ```
///
/// `no_run` test:
///
/// ```rust,no_run
/// assert!(!cargo_miri_test::make_true());
/// ```
///
/// `compile_fail` test:
///
/// ```rust,compile_fail
/// assert!(cargo_miri_test::make_true() == 5);
/// ```
///
/// Post-monomorphization error in `compile_fail` test:
///
/// ```rust,compile_fail
/// struct Fail<T>(T);
/// impl<T> Fail<T> {
///     const C: () = panic!();
/// }
///
/// let _val = Fail::<i32>::C;
/// ```
// This is imported in `main.rs`.
#[unsafe(no_mangle)]
pub fn make_true() -> bool {
    proc_macro_crate::use_the_dependency!();
    issue_1567::use_the_dependency();
    issue_1705::use_the_dependency();
    issue_1691::use_me()
}

/// ```rust
/// cargo_miri_test::miri_only_fn();
/// ```
#[cfg(miri)]
pub fn miri_only_fn() {}

pub fn main() {
    println!("imported main");
}
