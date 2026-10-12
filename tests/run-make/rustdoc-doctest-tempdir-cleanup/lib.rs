/// ```
/// assert!(true);
/// ```
pub fn a() {}

/// ```
/// let count = std::fs::read_dir(std::env::temp_dir())
///     .unwrap()
///     .filter(|entry| {
///         entry.as_ref().unwrap().file_name().to_str().unwrap().starts_with("rustdoctest")
///     })
///     .count();
/// // It's 2 and not one because rustdoc creates a temp dir where it puts the `rustdoc-cfgs`
/// // file which contains the list of flags to be passed down to the doctests. It lives
/// // until all doctests have been run.
/// assert_eq!(count, 2);
/// ```
pub fn b() {}
