//@ revisions: html_static html_non_static html_static_coverage html_non_static_coverage
//@ compile-flags: -Zunstable-options --output-format json
//@[html_static] compile-flags: --emit html-static-files
//@[html_static_coverage] compile-flags: --emit html-static-files --show-coverage
//@[html_non_static] compile-flags:  --emit html-non-static-files
//@[html_non_static_coverage] compile-flags:  --emit html-non-static-files --show-coverage

//[html_static]~? ERROR the `--emit=html-static-files` flag is not supported with `--output-format=json`
//[html_static_coverage]~? ERROR the `--emit=html-static-files` flag is not supported with `--output-format=json`
//[html_non_static]~? ERROR the `--emit=html-non-static-files` flag is not supported with `--output-format=json`
//[html_non_static_coverage]~? ERROR the `--emit=html-non-static-files` flag is not supported with `--output-format=json`
