// Regression test for issue #93210.

//@ aux-crate:doc_hidden_fields=doc-hidden-fields.rs
//@ edition: 2021

#[derive(Default)]
pub struct A {
    #[doc(hidden)]
    pub hello: i32,
    pub bye: i32,
}

#[derive(Default)]
pub struct C {
    pub hello: i32,
    pub bye: i32,
}

fn main() {
    // We want to list the field `hello` despite being marked
    // `doc(hidden)` because it's defined in this crate.
    A::default().hey;
    //~^ ERROR no field `hey` on type `A`
    //~| NOTE unknown field
    //~| NOTE available fields are: `hello`, `bye`

    // Here we want to hide the field `hello` since it's marked
    // `doc(hidden)` and comes from an external crate.
    doc_hidden_fields::B::default().hey;
    //~^ ERROR no field `hey` on type `B`
    //~| NOTE unknown field
    //~| NOTE available field is: `bye`

    C::default().hey;
    //~^ ERROR no field `hey` on type `C`
    //~| NOTE unknown field
    //~| NOTE available fields are: `hello`, `bye`
}
