/// ```
/// unsafe extern "C" {
///     fn cfoo() -> i32;
/// }
/// assert_eq!(unsafe { cfoo() }, 42);
/// ```
pub fn documented() {}
