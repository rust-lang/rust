#![deny(rustdoc::broken_intra_doc_links)]
//~^ NOTE lint level is defined
pub enum S {
    A,
}
fn S() {}

#[macro_export]
macro_rules! m {
    () => {};
}

static s: usize = 0;
const c: usize = 0;

trait T {}

struct X {
    y: usize,
}

/// Link to [struct@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with `enum@`

/// Link to [mod@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with `enum@`

/// Link to [union@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with `enum@`

/// Link to [trait@S]
//~^ ERROR incompatible link kind for `S`
//~| NOTE this link resolved
//~| HELP prefix with `enum@`

/// Link to [struct@T]
//~^ ERROR incompatible link kind for `T`
//~| NOTE this link resolved
//~| HELP prefix with `trait@`

/// Link to [derive@m]
//~^ ERROR incompatible link kind for `m`
//~| NOTE this link resolved
//~| HELP add an exclamation mark

/// Link to [m()]
//~^ ERROR unresolved link to `m`
//~| NOTE this link resolves to the macro `m`
//~| HELP add an exclamation mark
/// and to [m!()]

/// Link to [const@s]
//~^ ERROR incompatible link kind for `s`
//~| NOTE this link resolved
//~| HELP prefix with `static@`

/// Link to [static@c]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP prefix with `const@`

/// Link to [fn@c]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP prefix with `const@`

/// Link to [c()]
//~^ ERROR incompatible link kind for `c`
//~| NOTE this link resolved
//~| HELP prefix with `const@`

/// Link to [const@f]
//~^ ERROR incompatible link kind for `f`
//~| NOTE this link resolved
//~| HELP add parentheses

/// Link to [fn@std]
//~^ ERROR unresolved link to `std`
//~| NOTE this link resolves to the crate `std`
//~| HELP to link to the crate, prefix with `mod@`

/// Link to [method@X::y]
//~^ ERROR incompatible link kind for `X::y`
//~| NOTE this link resolved
//~| HELP prefix with `field@`

/// Link to [field@S::A]
//~^ ERROR unresolved link to `S::A`
//~| NOTE this link resolves
//~| HELP prefix with `variant@`
pub fn f() {}
