#![deny(broken_intra_doc_links)]
//~^ NOTE lint level is defined

/// [char]
//~^ ERROR both a module and a builtin type
//~| NOTE ambiguous link
//~| HELP to link to the module
//~| HELP to link to the builtin type

/// [type@char]
//~^ ERROR both a module and a builtin type
//~| NOTE ambiguous link
//~| HELP to link to the module
//~| HELP to link to the builtin type

/// [mod@char] // ok
/// [prim@char] // ok

/// [struct@char]
//~^ ERROR incompatible link
//~| HELP prefix with `mod@`
//~| NOTE resolved to a module
pub mod char {}

pub mod inner {
    //! [struct@char]
    //~^ ERROR incompatible link
    //~| HELP prefix with `prim@`
    //~| NOTE resolved to a builtin type
}
