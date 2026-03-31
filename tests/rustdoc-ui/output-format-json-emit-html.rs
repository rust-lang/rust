//@ revisions: html_static html_non_static
//@ check-fail
//@[html_static] compile-flags: -Z unstable-options --output-format=json --emit=html-static-files
//@[html_non_static] compile-flags: -Z unstable-options --output-format=json --emit=html-non-static-files
//[html_static]~? ERROR the `--emit=html-static-files` flag is not supported with `--output-format=json`
//[html_non_static]~? ERROR the `--emit=html-non-static-files` flag is not supported with `--output-format=json`

pub struct Foo;
