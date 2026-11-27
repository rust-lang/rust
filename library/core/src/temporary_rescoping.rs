//! Operators for modifying temporary scopes

/// `scope!('l => $expr)` sets the [enclosing temporary scope] of `$expr` to that of the
/// expression labeled with `'l`.
///
/// [enclosing temporary scope]: https://doc.rust-lang.org/reference/destructors.html#r-destructors.scope.temporary.enclosing
// FIXME(temporary_rescoping): add documentation, ideally using Reference terminology
#[allow_internal_unstable(internal_syntax)]
#[unstable(feature = "temporary_rescoping", issue = "139076")]
#[diagnostic::opaque]
pub macro scope($($arg:tt)*) {
    k#rescope_scope($($arg)*)
}

/// `extend!('l => $expr)` sets the [lifetime-extension context] of `$expr` to that of the
/// expression labeled with `'l`.
///
/// [lifetime-extension context]: https://doc.rust-lang.org/reference/destructors.html#temporary-lifetime-extension
// FIXME(temporary_rescoping): add documentation, ideally using Reference terminology
#[allow_internal_unstable(internal_syntax)]
#[unstable(feature = "temporary_rescoping", issue = "139076")]
#[diagnostic::opaque]
pub macro extend($($arg:tt)*) {
    k#rescope_extend($($arg)*)
}
