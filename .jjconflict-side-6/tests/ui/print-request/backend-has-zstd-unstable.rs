//! Check that `--print=backend-has-zstd` is unstable.
//!
//! That print value is intended for use by compiletest, and should probably
//! never be stabilized in this form.

//@ compile-flags: --print=backend-has-zstd

//~? ERROR: the `-Z unstable-options` flag must also be passed
