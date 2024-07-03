//@ aux-build:missing-assoc-fn-applicable-suggestions.rs

extern crate missing_assoc_fn_applicable_suggestions;
use missing_assoc_fn_applicable_suggestions::TraitA;

struct S;
impl TraitA<()> for S {
    //~^ ERROR not all trait items implemented
}
//~^ HELP implement the missing item
//~| HELP implement the missing item
//~| HELP implement the missing item
//~| HELP implement the missing item

fn main() {}
