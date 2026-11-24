#![feature(where_clause_attrs)]
#![allow(invalid_doc_attributes)]
#![warn(unused_doc_comments)]

fn test()
where
#[doc(alias = ":(")]
//~^ ERROR cannot be used on
//~| ERROR `#[doc(alias = "...")]` isn't allowed on where predicate
():,

#[doc(hidden)]
//~^ ERROR cannot be used on
():,

#[doc = ""]
//~^ ERROR unused doc comment
():,

#[doc()]
//~^ ERROR cannot be used on
():,

#[doc(5)]
//~^ ERROR cannot be used on
():,

#[doc]
//~^ ERROR cannot be used on
():,

#[doc = 5]
//~^ ERROR cannot be used on
():,

{ }

fn main() {}
