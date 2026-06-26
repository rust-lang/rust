//@ check-pass

#![feature(associated_type_defaults, where_clause_attrs)]
#![allow(deprecated_where_clause_location)]

trait A {
    type F<T>
    where
        #[cfg(false)]
        T: TraitB,
    = T;
}

fn main() {}
