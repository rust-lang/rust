// these are rustfixable, but run-rustfix tests cannot handle them

const VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR: Consider removing 'static

const VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];

static STATIC_VAR_FOUR: (&str, (&str, &'static str), &'static str) = ("on", ("th", "th"), "on"); // ERROR: Consider removing 'static

static STATIC_VAR_FIVE: &'static [&[&'static str]] = &[&["test"], &["other one"]]; // ERROR: Consider removing 'static

static STATIC_VAR_SEVEN: &[&(&str, &'static [&'static str])] = &[&("one", &["other one"])];

fn main() {}
