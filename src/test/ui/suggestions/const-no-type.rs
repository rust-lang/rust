// In the cases below, the type is missing from the `const` and `static` items.
//
// Here, we test that we:
//
// a) Perform parser recovery.
//
// b) Emit a diagnostic with the actual inferred type to RHS of `=` as the suggestion.

fn main() {}

// These will not reach typeck:

#[cfg(FALSE)]
const C2 = 42;
//~^ ERROR missing type for `const` item
//~| HELP provide a type for the item
//~| SUGGESTION C2: <type>

#[cfg(FALSE)]
static S2 = "abc";
//~^ ERROR missing type for `static` item
//~| HELP provide a type for the item
//~| SUGGESTION S2: <type>

#[cfg(FALSE)]
static mut SM2 = "abc";
//~^ ERROR missing type for `static mut` item
//~| HELP provide a type for the item
//~| SUGGESTION SM2: <type>

// These will, so the diagnostics should be stolen by typeck:

const C = 42;
//~^ ERROR missing type for `const` item
//~| HELP provide a type for the constant
//~| SUGGESTION C: i32

const D = &&42;
//~^ ERROR missing type for `const` item
//~| HELP provide a type for the constant
//~| SUGGESTION D: &&i32

static S = Vec::<String>::new();
//~^ ERROR missing type for `static` item
//~| HELP provide a type for the static variable
//~| SUGGESTION S: Vec<String>

static mut SM = "abc";
//~^ ERROR missing type for `static mut` item
//~| HELP provide a type for the static variable
//~| SUGGESTION &str
