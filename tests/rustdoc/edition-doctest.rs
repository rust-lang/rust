//@ compile-flags:--test

/// ```rust,edition2018
/// #![feature(try_blocks)]
///
/// use std::num::ParseIntError;
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "2".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert_eq!(result, Ok(6));
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "foo".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert!(result.is_err());
/// ```


/// ```rust,edition2015,compile_fail,E0574
/// #![feature(try_blocks)]
///
/// use std::num::ParseIntError;
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "2".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert_eq!(result, Ok(6));
///
/// let result: Result<i32, ParseIntError> = try {
///     "1".parse::<i32>()?
///         + "foo".parse::<i32>()?
///         + "3".parse::<i32>()?
/// };
/// assert!(result.is_err());
/// ```

pub fn foo() {}
