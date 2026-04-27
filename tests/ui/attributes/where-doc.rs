#![feature(where_clause_attrs)]
#![allow(invalid_doc_attributes)]

fn test()
where
#[doc(alias = ":(")]
//~^ ERROR most attributes are not supported in `where` clauses
//~| ERROR `#[doc(alias = "...")]` isn't allowed on where predicate
():,

#[doc(hidden)]
//~^ ERROR most attributes are not supported in `where` clauses
():,

#[doc = ""]
//~^ ERROR most attributes are not supported in `where` clauses
():,

{ }

fn main() {}
