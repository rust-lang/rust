// https://github.com/rust-lang/rust/issues/31899
#![crate_name="issue_31899"]

//@ has issue_31899/index.html
//@ hasraw - 'Make this line a bit longer.'
//@ !hasraw - 'rust rust-example-rendered'
//@ !hasraw - 'use ndarray::arr2'
//@ !hasraw - 'prohibited'

/// A tuple or fixed size array that can be used to index an array.
/// Make this line a bit longer.
///
/// ```
/// use ndarray::arr2;
///
/// let mut a = arr2(&[[0, 1], [0, 0]]);
/// a[[1, 1]] = 1;
/// assert_eq!(a[[0, 1]], 1);
/// assert_eq!(a[[1, 1]], 1);
/// ```
///
/// **Note** the blanket implementation that's not visible in rustdoc:
/// `impl<D> NdIndex for D where D: Dimension { ... }`
pub fn bar() {}

/// Some line
///
/// # prohibited
pub fn foo() {}

/// Some line
///
/// 1. prohibited
/// 2. bar
pub fn baz() {}

/// Some line
///
/// - prohibited
/// - bar
pub fn qux() {}

/// Some line
///
/// * prohibited
/// * bar
pub fn quz() {}

/// Some line
///
/// > prohibited
/// > bar
pub fn qur() {}

/// Some line
///
/// prohibited
/// =====
///
/// Second
/// ------
pub fn qut() {}
