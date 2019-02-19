// compile-flags:--test

/// A check of using various process termination strategies
///
/// # Examples
///
/// ```rust
/// assert!(true); // this returns `()`, all is well
/// ```
///
/// You can also simply return `Ok(())`, but you'll need to disambiguate the
/// type using turbofish, because we cannot infer the type:
///
/// ```rust
/// Ok::<(), &'static str>(())
/// ```
///
/// You can err with anything that implements `Debug`:
///
/// ```rust,should_panic
/// Err("This is returned from `main`, leading to panic")?;
/// Ok::<(), &'static str>(())
/// ```
pub fn check_process_termination() {}
