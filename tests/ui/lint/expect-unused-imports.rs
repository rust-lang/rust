//@ check-pass

#![deny(unused_imports)]
#![deny(unfulfilled_lint_expectations)]

#[expect(unused_imports)]
use std::{io, fs};

fn main() {}
