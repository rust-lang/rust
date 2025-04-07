//! Check that we point to `-Whelp` to guide the user to find the list of lints if the user requests
//! `--print=lints` (which is not a valid print request).

//@ compile-flags: --print lints
//@ error-pattern: help: use `-Whelp` to print a list of lints
//@ error-pattern: help: for more information, see the rustc book

//~? ERROR unknown print request: `lints`
