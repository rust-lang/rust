//! Ensure that we check the predicates at the definition site for well-formedness.
#![feature(lazy_type_alias)]
#![allow(incomplete_features)]

type Alias0 = ()
where
    Vec<str>:; //~ ERROR the size for values of type `str` cannot be known at compilation time

type Alias1 = ()
where
    Vec<str>: Sized; //~ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
