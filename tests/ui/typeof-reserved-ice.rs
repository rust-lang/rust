// check-pass
// compile-flags: --error-format=short
// Regression test for issue #147146
// `typeof` is reserved but not implemented.
// This should produce a normal error, not ICE.

impl typeof(|| {}) {}
