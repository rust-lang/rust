/// ```
/// assert_eq!(doctest::add(2, 2), 4);
/// ```
///
/// ```should_panic
/// assert_eq!(doctest::add(2, 2), 5);
/// ```
///
/// ```compile_fail
/// assert_eq!(doctest::add(2, 2), "banana");
/// ```
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}
