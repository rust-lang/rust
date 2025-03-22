#![feature(where_clause_attrs, cfg_boolean_literals, lazy_type_alias)]
#![expect(incomplete_features)]

struct Foo;
trait Trait {}

impl Trait for Foo {}

type MixedWhereBounds1
where //~ ERROR: where clauses are not allowed before the type for type aliases
    #[cfg(true)]
    Foo: Trait,
= Foo
where
    (): Sized;

type MixedWhereBounds2
where //~ ERROR: where clauses are not allowed before the type for type aliases
    #[cfg(false)]
    Foo: Trait,
= Foo
where
    (): Sized;

type MixedWhereBounds3
where
//~^ ERROR: where clauses are not allowed before the type for type aliases
    Foo: Trait,
= Foo
where
    (): Sized;

fn main() {}
