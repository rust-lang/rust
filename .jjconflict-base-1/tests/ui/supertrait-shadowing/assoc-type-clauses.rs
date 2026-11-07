//@ normalize-stderr: "assoc_type_clauses\[[^\]]+\]" -> "assoc_type_clauses[HASH]"

#![feature(rustc_attrs)]
#![feature(supertrait_item_shadowing)]
#![allow(dead_code)]

trait A {
    type Assoc;
}
impl<T> A for T {
    type Assoc = i8;
}

trait B: A {
    type Assoc;
}
impl<T> B for T {
    type Assoc = i16;
}

trait C: B {}
impl<T> C for T {}

#[rustc_dump_clauses]
fn a_bound<T: A<Assoc = i8>>() {}
//~^ ERROR rustc_dump_clauses
//~| NOTE TraitClause(<T as std::marker::Sized>
//~| NOTE TraitClause(<T as A>
//~| NOTE A::Assoc

#[rustc_dump_clauses]
fn b_bound<T: B<Assoc = i16>>() {}
//~^ ERROR rustc_dump_clauses
//~| NOTE TraitClause(<T as std::marker::Sized>
//~| NOTE TraitClause(<T as B>
//~| NOTE B::Assoc

#[rustc_dump_clauses]
fn c_bound<T: C<Assoc = i16>>() {}
//~^ ERROR rustc_dump_clauses
//~| NOTE TraitClause(<T as std::marker::Sized>
//~| NOTE TraitClause(<T as C>
//~| NOTE B::Assoc

fn main() {}
