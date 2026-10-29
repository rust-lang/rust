// rustfmt-format_code_in_doc_comments: true

/// Should format
/// ```rust
/// assert!( false );
/// ```
///
/// Should format
/// ```rust,should_panic
/// assert!( false );
/// ```
///
/// Should format
/// ```rust,should_panic,edition2018
/// assert!( false );
/// ```
///
/// Should format
/// ```rust , should_panic , edition2018
/// assert!( false );
/// ```
///
/// Should not format
/// ```ignore
/// assert!( false );
/// ```
///
/// Should not format (not all are rust)
/// ```rust,ignore
/// assert!( false );
/// ```
///
/// Should not format (rust compile_fail)
/// ```compile_fail
/// assert!( false );
/// ```
///
/// Should not format (rust compile_fail)
/// ```rust,compile_fail
/// assert!( false );
/// ```
///
/// Various unspecified ones that should format
/// ```
/// assert!( false );
/// ```
///
/// ```,
/// assert!( false );
/// ```
///
/// ```,,,,,
/// assert!( false );
/// ```
///
/// ```,,,  rust  ,,
/// assert!( false );
/// ```
///
/// Should not format
/// ```,,,  rust  ,  ignore,
/// assert!( false );
/// ```
///
/// Few empty ones
/// ```
/// ```
///
/// ```rust
/// ```
///
/// ```ignore
/// ```
fn foo() {}
