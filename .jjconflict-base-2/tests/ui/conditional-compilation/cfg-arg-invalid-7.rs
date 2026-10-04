// Regression test for issue #89358.

//@ edition: 2015..2021
//@ compile-flags: --cfg a"

//~? RAW unterminated double quote string
//~? RAW this occurred on the command line
