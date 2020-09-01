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
//~| HELP prefix with the item kind

/// Link to [mod@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [union@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [trait@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [struct@T]
//~^ ERROR incompatible link kind for `T`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [derive@m]
//~^ ERROR incompatible link kind for `m`
//~| NOTE this link resolved
//~| HELP add an exclamation mark

/// Link to [const@s]
//~^ ERROR incompatible link kind for `s`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [static@c]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [fn@c]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [c()]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP prefix with the item kind

/// Link to [const@f]
//~^ ERROR incompatible link kind for `f`
//~| NOTE this link resolved
//~| HELP add parentheses
pub fn f() {}
