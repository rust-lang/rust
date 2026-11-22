#![feature(where_clause_attrs)]

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

// == That the doc attributes below don't trigger the error is a bug
#[doc()]
//~^ ERROR
():,

#[doc(5)]
//~^ ERROR
():,

#[doc]
//~^ ERROR
():,

#[doc = 5]
//~^ ERROR
():,

{ }

fn main() {}
