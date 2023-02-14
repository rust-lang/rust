#![feature(associated_type_bounds)]

trait B {
    type AssocType;
}

fn f()
where
    dyn for<'j> B<AssocType: 'j>:,
    //~^ ERROR associated type bounds are only allowed in where clauses and function signatures
    //~| ERROR the value of the associated type `AssocType` (from trait `B`) must be specified
{
}

fn main() {}
