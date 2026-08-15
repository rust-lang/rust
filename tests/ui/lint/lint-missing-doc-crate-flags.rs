// This test checks that we lint on the crate when it's missing a documentation.
//
//@ compile-flags: -Dmissing-docs --crate-type=lib
//~ ERROR missing documentation for the crate
