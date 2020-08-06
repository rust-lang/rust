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
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [mod@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [union@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [trait@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [struct@T]
//~^ ERROR incompatible link kind for `T`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [derive@m]
//~^ ERROR incompatible link kind for `m`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [const@s]
//~^ ERROR incompatible link kind for `s`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [static@c]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [fn@c]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [c()]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP use its disambiguator

/// Link to [const@f]
//~^ ERROR incompatible link kind for `f`
//~| NOTE this link resolved
//~| HELP use its disambiguator
pub fn f() {}
