#![feature(custom_inner_attributes)]
#![rustfmt::skip] // use_foo must be referenced before foo

// Load foo-v2 through use-foo
use use_foo as _;

// Make sure we don't disambiguate this as foo-v2.
use foo as _;

fn main() {}
