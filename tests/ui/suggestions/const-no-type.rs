// In the cases below, the type is missing from the `const` and `static` items.
//
// Here, we test that we:
//
// a) Perform parser recovery.
//
// b) Emit a diagnostic with the actual inferred type to RHS of `=` as the suggestion.

fn main() {}

// These will not reach typeck:

#[cfg(false)]
const C2 = 42;
//~^ ERROR: omitting type on const item declaration is experimental [E0658]
//~| HELP: add `#![feature(const_items_unit_type_default)]` to the crate attributes to enable
//~| HELP: consider specifying the type explicitly

#[cfg(false)]
static S2 = "abc";
//~^ ERROR missing type for `static` item
//~| HELP provide a type for the item
//~| SUGGESTION : <type>

#[cfg(false)]
static mut SM2 = "abc";
//~^ ERROR missing type for `static mut` item
//~| HELP provide a type for the item
//~| SUGGESTION : <type>

// These will, so the diagnostics should be stolen by typeck:

const C = 42;
//~^ ERROR: omitting type on const item declaration is experimental [E0658]
//~| HELP: add `#![feature(const_items_unit_type_default)]` to the crate attributes to enable
//~| HELP: consider specifying the type explicitly
//~| ERROR: mismatched types [E0308]

const D = &&42;
//~^ ERROR: omitting type on const item declaration is experimental [E0658]
//~| HELP: add `#![feature(const_items_unit_type_default)]` to the crate attributes to enable
//~| HELP: consider specifying the type explicitly
//~| ERROR: mismatched types [E0308]

static S = Vec::<String>::new();
//~^ ERROR missing type for `static` item
//~| HELP provide a type for the static variable
//~| SUGGESTION : Vec<String>

static mut SM = "abc";
//~^ ERROR missing type for `static mut` item
//~| HELP provide a type for the static variable
//~| SUGGESTION : &str
