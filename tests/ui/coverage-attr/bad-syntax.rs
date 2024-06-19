#![feature(coverage_attribute)]

// Tests the error messages produced (or not produced) by various unusual
// uses of the `#[coverage(..)]` attribute.

// FIXME(#84605): Multiple coverage attributes with the same value are useless,
// and should probably produce a diagnostic.
#[coverage(off)]
#[coverage(off)]
fn multiple_consistent() {}

// FIXME(#84605): When there are multiple inconsistent coverage attributes,
// it's unclear which one will prevail.
#[coverage(off)]
#[coverage(on)]
fn multiple_inconsistent() {}

#[coverage] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn bare_word() {}

// FIXME(#84605): This shows as multiple different errors, one of which suggests
// writing bare `#[coverage]`, which is not allowed.
#[coverage = true]
//~^ ERROR expected `coverage(off)` or `coverage(on)`
//~| ERROR malformed `coverage` attribute input
//~| HELP the following are the possible correct uses
//~| SUGGESTION #[coverage(on|off)]
fn key_value() {}

#[coverage()] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn list_empty() {}

#[coverage(off, off)] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn list_consistent() {}

#[coverage(off, on)] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn list_inconsistent() {}

#[coverage(bogus)] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn bogus_word() {}

#[coverage(bogus, off)] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn bogus_word_before() {}

#[coverage(off, bogus)] //~ ERROR expected `coverage(off)` or `coverage(on)`
fn bogus_word_after() {}

#[coverage(off,)]
fn comma_after() {}

// FIXME(#84605): This shows as multiple different errors.
#[coverage(,off)]
//~^ ERROR expected identifier, found `,`
//~| HELP remove this comma
//~| ERROR expected `coverage(off)` or `coverage(on)`
fn comma_before() {}

fn main() {}
