//! Ensure that when a macro (or normal code) does `#[deny]` inside a `#[forbid]` context, no error
//! is emitted, as both parties agree on the treatment of the lint.
//!
//! However, still emit an error if the macro does `#[allow]` or `#[warn]`.

//@ revisions: forbid deny warn allow
//@[forbid] aux-build:forbid-macro.rs
//@[deny] aux-build:deny-macro.rs
//@[warn] aux-build:warn-macro.rs
//@[allow] aux-build:allow-macro.rs

//@[forbid] check-pass
//@[deny] check-pass

#![forbid(unsafe_code)]

#[cfg(allow)]
extern crate allow_macro;
#[cfg(deny)]
extern crate deny_macro;
#[cfg(forbid)]
extern crate forbid_macro;
#[cfg(warn)]
extern crate warn_macro;

fn main() {
    #[cfg(forbid)]
    forbid_macro::emit_forbid! {} // OK

    #[cfg(deny)]
    deny_macro::emit_deny! {} // OK

    #[cfg(warn)]
    warn_macro::emit_warn! {}
    //[warn]~^ ERROR warn(unsafe_code) incompatible with previous forbid
    //[warn]~| ERROR warn(unsafe_code) incompatible with previous forbid

    #[cfg(allow)]
    allow_macro::emit_allow! {}
    //[allow]~^ ERROR allow(unsafe_code) incompatible with previous forbid
    //[allow]~| ERROR allow(unsafe_code) incompatible with previous forbid

    #[deny(unsafe_code)] // OK
    let _ = 0;
}
