// Test that we do not suggest to add type annotations for unnamable types.

#![crate_type="lib"]
#![feature(generators)]

const A = 5;
//~^ ERROR: missing type for `const` item
//~| HELP: provide a type for the item

static B: _ = "abc";
//~^ ERROR: the type placeholder `_` is not allowed within types on item signatures
//~| NOTE: not allowed in type signatures
//~| HELP: replace with the correct type


const C: _ = || 42;
//~^ ERROR: the type placeholder `_` is not allowed within types on item signatures
//~| NOTE: not allowed in type signatures
//~| NOTE: however, the inferred type

struct S<T> { t: T }
const D = S { t: || -> i32 { 42 } };
//~^ ERROR: missing type for `const` item
//~| NOTE: however, the inferred type

fn foo() -> i32 { 42 }
const E = S { t: foo };
//~^ ERROR: missing type for `const` item
//~| NOTE: however, the inferred type

const F = || -> i32 { yield 0; return 1; };
//~^ ERROR: missing type for `const` item
//~| NOTE: however, the inferred type
