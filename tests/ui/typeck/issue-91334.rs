// Regression test for the ICE described in issue #91334.

//@error-in-other-file: this file contains an unclosed delimiter

#![feature(generators)]

fn f(){||yield(((){),
