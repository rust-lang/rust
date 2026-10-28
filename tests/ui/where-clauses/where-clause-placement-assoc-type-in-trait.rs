// Ensure that we permit leading where clauses on associated types in traits
// (for backward compatibility) and lint on them.

//@ check-pass
//@ run-rustfix

#![feature(associated_type_defaults)] // only used for some test cases
#![allow(dead_code)]

trait Trait {
    // FIXME(fmease): Emit `deprecated_where_clause_location` here, too:
    type Assoc0<T> where T: Copy where T: Eq;

    type Assoc1 where u32: Copy = ();
    //~^ WARNING where clause not allowed here

    type Assoc2 where u32: Copy = () where i32: Copy;
    //~^ WARNING where clause not allowed here
}

fn main() {}
