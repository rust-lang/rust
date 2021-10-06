The compiler found multiple library files with the requested crate name.

This error can occur in several different cases -- for example, when using
`extern crate` or passing `--extern` options without crate paths. It can also be
caused by caching issues with the build directory, in which case `cargo clean`
may help.
