//@ run-rustfix

#![feature(lazy_type_alias)]
#![allow(incomplete_features)]
#![crate_type = "lib"]

// Check that we *reject* leading where-clauses on lazy type aliases.

pub type Leading0<T>
where //~ ERROR where clauses are not allowed before the type for type aliases
    String: From<T>,
= T;

pub type Leading1<T, U>
where //~ ERROR where clauses are not allowed before the type for type aliases
    String: From<T>,
= (T, U)
where
    U: Copy;

pub type EmptyLeading0 where = ();
//~^ ERROR where clauses are not allowed before the type for type aliases

pub type EmptyLeading1<T> where = T where T: Copy;
//~^ ERROR where clauses are not allowed before the type for type aliases
