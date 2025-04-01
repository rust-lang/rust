// Test that we do not suggest to add type annotations for unnamable types.

#![crate_type="lib"]
#![feature(coroutines, stmt_expr_attributes, const_async_blocks)]

const A = 5;
//~^ ERROR: missing type for `const` item
//~| HELP: provide a type for the constant

static B: _ = "abc";
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for static variables
//~| NOTE: not allowed in type signatures
//~| HELP: replace this with a fully-specified type


// FIXME: this should also suggest a function pointer, as the closure is non-capturing
const C: _ = || 42;
//~^ ERROR: the placeholder `_` is not allowed within types on item signatures for constants
//~| NOTE: not allowed in type signatures
//~| NOTE: however, the inferred type

struct S<T> { t: T }
const D = S { t: { let i = 0; move || -> i32 { i } } };
//~^ ERROR: missing type for `const` item
//~| NOTE: however, the inferred type


fn foo() -> i32 { 42 }
const E = foo;
//~^ ERROR: missing type for `const` item
//~| HELP: provide a type for the constant
const F = S { t: foo };
//~^ ERROR: missing type for `const` item
//~| HELP: provide a type for the constant


const G = #[coroutine] || -> i32 { yield 0; return 1; };
//~^ ERROR: missing type for `const` item
//~| NOTE: however, the inferred type
