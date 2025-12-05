#![feature(coverage_attribute)]
//@ edition: 2021
//@ reference: attributes.coverage.syntax
//@ reference: attributes.coverage.duplicates

// Tests the error messages produced (or not produced) by various unusual
// uses of the `#[coverage(..)]` attribute.

#[coverage(off)] //~ ERROR multiple `coverage` attributes
#[coverage(off)]
fn multiple_consistent() {}

#[coverage(off)] //~ ERROR multiple `coverage` attributes
#[coverage(on)]
fn multiple_inconsistent() {}

#[coverage] //~ ERROR malformed `coverage` attribute input
fn bare_word() {}

#[coverage = true] //~ ERROR malformed `coverage` attribute input
fn key_value() {}

#[coverage()] //~ ERROR malformed `coverage` attribute input
fn list_empty() {}

#[coverage(off, off)] //~ ERROR malformed `coverage` attribute input
fn list_consistent() {}

#[coverage(off, on)] //~ ERROR malformed `coverage` attribute input
fn list_inconsistent() {}

#[coverage(bogus)] //~ ERROR malformed `coverage` attribute input
fn bogus_word() {}

#[coverage(bogus, off)] //~ ERROR malformed `coverage` attribute input
fn bogus_word_before() {}

#[coverage(off, bogus)] //~ ERROR malformed `coverage` attribute input
fn bogus_word_after() {}

#[coverage(off,)] // (OK!)
fn comma_after() {}

#[coverage(,off)] //~ ERROR expected identifier, found `,`
fn comma_before() {}

fn main() {}
