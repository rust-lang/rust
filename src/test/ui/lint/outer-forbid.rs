// Forbidding a group (here, `unused`) overrules subsequent allowance of both
// the group, and an individual lint in the group (here, `unused_variables`);
// and, forbidding an individual lint (here, `non_snake_case`) overrules
// subsequent allowance of a lint group containing it (here, `nonstandard_style`). See
// Issue #42873.

// If you turn off deduplicate diagnostics (which rustc turns on by default but
// compiletest turns off when it runs ui tests), then the errors are
// (unfortunately) repeated here because the checking is done as we read in the
// errors, and currently that happens two or three different times, depending on
// compiler flags.
//
// The test is much cleaner if we deduplicate, though.

// compile-flags: -Z deduplicate-diagnostics=yes

#![forbid(unused, non_snake_case)]

#[allow(unused_variables)] //~ ERROR incompatible with previous
fn foo() {}

#[allow(unused)] //~ ERROR incompatible with previous
fn bar() {}

#[allow(nonstandard_style)] //~ ERROR incompatible with previous
fn main() {
    println!("hello forbidden world")
}
