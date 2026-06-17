// FIXME(lazy_type_alias): Improve this test.
//@ check-pass

#![feature(auto_traits)]

auto trait Marker {}

struct Local;
type Alias = Local;

impl Marker for Alias {}

fn main() {}
