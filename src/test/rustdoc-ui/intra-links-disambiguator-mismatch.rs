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
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [mod@S]
//~^ ERROR unresolved link to `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [union@S]
//~^ ERROR unresolved link to `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [trait@S]
//~^ ERROR unresolved link to `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [struct@T]
//~^ ERROR unresolved link to `T`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [derive@m]
//~^ ERROR unresolved link to `m`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [const@s]
//~^ ERROR unresolved link to `s`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [static@c]
//~^ ERROR unresolved link to `c`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [fn@c]
//~^ ERROR unresolved link to `c`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [c()]
//~^ ERROR unresolved link to `c`
//~| NOTE this link resolved
//~| HELP use its disambiguator
pub fn f() {}
