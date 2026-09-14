//@ compile-flags: -Z unstable-options --show-coverage --output-format=json --emit=html-non-static-files
//@ check-fail
//~? ERROR the `--emit=html-non-static-files` flag is not supported with `--output-format=json`

pub struct Foo;
