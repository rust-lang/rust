// FIXME: Rewrite into proper test.
//@ check-pass

#![deny(missing_debug_implementations)]
#![feature(lazy_type_alias)]

pub struct Local;

type Alias = Local;

impl std::fmt::Debug for Alias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Local")
    }
}

fn main() {}
