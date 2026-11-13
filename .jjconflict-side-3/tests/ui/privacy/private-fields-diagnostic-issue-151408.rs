//@ aux-build:private-fields-diagnostic-aux-issue-151408.rs

extern crate private_fields_diagnostic_aux_issue_151408 as aux;

use aux::{Named, NamedWithMultipleFields, PublicTuple};

fn main() {
    let _ = Named {};
    //~^ ERROR cannot construct `aux::Named` with struct literal syntax due to private fields

    let _ = PublicTuple();
    //~^ ERROR cannot initialize a tuple struct which contains private fields [E0423]

    // Keep the private-field note when the user already wrote part of the struct literal.
    let _ = NamedWithMultipleFields { visible: 1 };
    //~^ ERROR cannot construct `NamedWithMultipleFields` with struct literal syntax due to private fields

    let _ = NamedWithMultipleFields {};
    //~^ ERROR cannot construct `NamedWithMultipleFields` with struct literal syntax due to private fields
}
