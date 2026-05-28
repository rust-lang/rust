//@ aux-build:missing-assoc-fn-applicable-suggestions.rs

extern crate missing_assoc_fn_applicable_suggestions;
use missing_assoc_fn_applicable_suggestions::TraitA;

struct S;
impl TraitA<()> for S {
    //~^ ERROR not all trait items implemented
}
//~^ HELP implement the missing item: `type Type = /* Type */;`
//~| HELP implement the missing item: `fn bar<T>(_: T) -> Self { todo!() }`
//~| HELP implement the missing item: `fn baz<T>(_: T) -> Self where T: TraitB, <T as TraitB>::Item: Copy { todo!() }`
//~| HELP implement the missing item: `const A: usize = 42;`

fn main() {}
