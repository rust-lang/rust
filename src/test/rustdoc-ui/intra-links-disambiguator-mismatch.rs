#![deny(broken_intra_doc_links)]
//~^ NOTE lint level is defined
pub enum S {}

macro_rules! m {
    () => {};
}

static s: usize = 0;
const c: usize = 0;

trait T {}

/// Link to [struct@S]
//~^ ERROR unresolved link to `S`
//~| NOTE did not match

/// Link to [mod@S]
//~^ ERROR unresolved link to `S`
//~| NOTE did not match

/// Link to [union@S]
//~^ ERROR unresolved link to `S`
//~| NOTE did not match

/// Link to [trait@S]
//~^ ERROR unresolved link to `S`
//~| NOTE did not match

/// Link to [struct@T]
//~^ ERROR unresolved link to `T`
//~| NOTE did not match

/// Link to [derive@m]
//~^ ERROR unresolved link to `m`
//~| NOTE did not match

/// Link to [const@s]
//~^ ERROR unresolved link to `s`
//~| NOTE did not match

/// Link to [static@c]
//~^ ERROR unresolved link to `c`
//~| NOTE did not match

/// Link to [fn@c]
//~^ ERROR unresolved link to `c`
//~| NOTE did not match

/// Link to [c()]
//~^ ERROR unresolved link to `c`
//~| NOTE did not match
pub fn f() {}
