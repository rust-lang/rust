// Test the parsing & basic semantic validation of where clauses on (free) type aliases.
//@ revisions: pass fail
//@[pass] check-pass

type Leading0 where u32: Copy = (); //[pass]~ WARN [type_alias_bounds]
type Leading1 where = ();

// Syntactically valid, semantically invalid:

#[cfg(fail)] type Trailing0 = () where u32: Copy;
//[fail]~^ ERROR where clauses are not allowed
#[cfg(fail)] type Trailing1 = () where;
//[fail]~^ ERROR where clauses are not allowed

// Indeed, we intentionally allow two where-clauses *at once* for backward compatibility.

#[cfg(fail)] type Both0 where (): = bool where !: Sized;
//[fail]~^ ERROR where clauses are not allowed
#[cfg(fail)] type Both1 where = bool where;
//[fail]~^ ERROR where clauses are not allowed

// We once used to incorrectly reject `BodilessBoth0` due to overzealous parse error recovery code.
#[cfg(false)] type BodilessBoth0 where [(); 1]: Copy where String: Clone;
#[cfg(false)] type BodilessBoth1 where where;

fn main() {}
