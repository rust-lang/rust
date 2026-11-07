// Demonstrate that decl macros can tell apart forced keywords from other kinds
// of identifiers when matching.
//
//@ edition: 2021..
#![feature(forced_keywords)]

macro_rules! accept {
    (ForcedKeyword k#type) => {};
    (Normal type) => {};
}

accept!(ForcedKeyword type); //~ ERROR no rules expected keyword `type`
accept!(ForcedKeyword r#type); //~ ERROR no rules expected `r#type`
accept!(ForcedKeyword k#type); // OK

accept!(Normal k#type); //~ ERROR no rules expected keyword `k#type`

fn main() {}
