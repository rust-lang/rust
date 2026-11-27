//! Check that we point to `-Whelp` to guide the user to find the list of lints if the user requests
//! `--print=lints` (which is not a valid print request).

//@ compile-flags: --print lints

//~? ERROR unknown print request: `lints`
//~? HELP use `-Whelp` to print a list of lints
//~? HELP for more information, see the rustc book
//~? HELP valid print requests are
