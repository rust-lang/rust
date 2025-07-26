// Make sure that we don't parse `extern crate async` as
// the front matter of a function leading us astray.

//@ edition: 2015
//@ check-pass

fn main() {}

#[cfg(false)]
extern crate async;

#[cfg(false)]
extern crate async as something_else;
