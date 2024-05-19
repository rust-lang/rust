//@no-rustfix: overlapping suggestions
// these are rustfixable, but run-rustfix tests cannot handle them

const VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR: Consider removing 'static
//~^ ERROR: constants have by default a `'static` lifetime
//~| NOTE: `-D clippy::redundant-static-lifetimes` implied by `-D warnings`
//~| ERROR: constants have by default a `'static` lifetime

const VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];
//~^ ERROR: constants have by default a `'static` lifetime
//~| ERROR: constants have by default a `'static` lifetime

static STATIC_VAR_FOUR: (&str, (&str, &'static str), &'static str) = ("on", ("th", "th"), "on"); // ERROR: Consider removing 'static
//~^ ERROR: statics have by default a `'static` lifetime
//~| ERROR: statics have by default a `'static` lifetime

static STATIC_VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR: Consider removing 'static
//~^ ERROR: statics have by default a `'static` lifetime
//~| ERROR: statics have by default a `'static` lifetime

static STATIC_VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];
//~^ ERROR: statics have by default a `'static` lifetime
//~| ERROR: statics have by default a `'static` lifetime

fn main() {}
