// It is unclear whether a fully unconfigured crate should link to standard library,
// or what its `no_std`/`no_core`/`compiler_builtins` status, more precisely.
// Currently the usual standard library prelude is added to such crates,
// and therefore they link to libstd.

#![cfg(FALSE)]
