//@ dont-check-compiler-stdout
//@ dont-check-compiler-stderr

//@ revisions: empty_format_with_colon
//@[empty_format_with_colon] compile-flags: --print=code-models:

//@ revisions: default_format
//@[default_format] compile-flags: --print=code-models
//@[default_format] check-pass

//@ revisions: non_existent_format
//@[non_existent_format] compile-flags: --print=code-models:diagram

//@ revisions: empty_format_with_colon_mnemonic
//@[empty_format_with_colon_mnemonic] compile-flags: --print=backend-has-mnemonic:RET: -Z unstable-options

//@ revisions: default_format_mnemonic
//@[default_format_mnemonic] compile-flags: --print=backend-has-mnemonic:RET -Z unstable-options
//@[default_format_mnemonic] check-pass

//@ revisions: non_existent_format_mnemonic
//@[non_existent_format_mnemonic] compile-flags: --print=backend-has-mnemonic:RET:diagram -Z unstable-options

fn main() {}

//[empty_format_with_colon]~? ERROR output format in --print is empty. remove the colon or provide a valid output format
//[non_existent_format]~? ERROR unsupported output format in --print: diagram

//[empty_format_with_colon_mnemonic]~? ERROR output format in --print is empty. remove the colon or provide a valid output format
//[non_existent_format_mnemonic]~? ERROR unsupported output format in --print: diagram
