//! Ensure that we check the predicates for well-formedness at the definition site.
#![feature(generic_const_items)]
#![expect(incomplete_features)]

const _: () = ()
where
    Vec<str>: Sized; //~ ERROR the size for values of type `str` cannot be known at compilation time

fn main() {}
