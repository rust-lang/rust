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
//~^ ERROR missing type for `const` item
//~| HELP provide a type for the item
//~| SUGGESTION : <type>

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
//~^ ERROR missing type for `const` item
//~| HELP provide a type for the constant
//~| SUGGESTION : i32

const D = &&42;
//~^ ERROR missing type for `const` item
//~| HELP provide a type for the constant
//~| SUGGESTION : &&i32

static S = Vec::<String>::new();
//~^ ERROR missing type for `static` item
//~| HELP provide a type for the static variable
//~| SUGGESTION : Vec<String>

static mut SM = "abc";
//~^ ERROR missing type for `static mut` item
//~| HELP provide a type for the static variable
//~| SUGGESTION : &str
