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
///
/// This also works with `Option<()>`s now:
///
/// ```rust
/// Some(())
/// ```
///
/// ```rust,should_panic
/// let x: &[u32] = &[];
/// let _ = x.iter().next()?;
/// Some(())
/// ```
pub fn check_process_termination() {}
