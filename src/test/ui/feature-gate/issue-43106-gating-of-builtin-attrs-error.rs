//~ NOTE: not an `extern crate` item
//~^ NOTE: not a function or static
//~^^ NOTE: not a function or closure
// This is testing whether various builtin attributes signals an
// error or warning when put in "weird" places.
//
// (This file sits on its own because it actually signals an error,
// which would mess up the treatment of other cases in
// issue-43106-gating-of-builtin-attrs.rs)

// ignore-tidy-linelength

#![macro_export]
//~^ ERROR: `macro_export` attribute cannot be used at crate level
#![main]
//~^ ERROR: `main` attribute cannot be used at crate level
#![start]
//~^ ERROR: `start` attribute cannot be used at crate level
#![repr()]
//~^ ERROR: `repr` attribute cannot be used at crate level
#![path = "3800"]
//~^ ERROR: `path` attribute cannot be used at crate level
#![automatically_derived]
//~^ ERROR: `automatically_derived` attribute cannot be used at crate level
#![no_mangle]
#![no_link]
//~^ ERROR: attribute should be applied to an `extern crate` item
#![export_name = "2200"]
//~^ ERROR: attribute should be applied to a function or static
#![inline]
//~^ ERROR: attribute should be applied to function or closure
#[inline]
//~^ ERROR attribute should be applied to function or closure
mod inline {
    //~^ NOTE not a function or closure

    mod inner { #![inline] }
    //~^ ERROR attribute should be applied to function or closure
    //~| NOTE not a function or closure

    #[inline = "2100"] fn f() { }
    //~^ ERROR attribute must be of the form
    //~| WARN this was previously accepted
    //~| NOTE #[deny(ill_formed_attribute_input)]` on by default
    //~| NOTE for more information, see issue #57571 <https://github.com/rust-lang/rust/issues/57571>

    #[inline] struct S;
    //~^ ERROR attribute should be applied to function or closure
    //~| NOTE not a function or closure

    #[inline] type T = S;
    //~^ ERROR attribute should be applied to function or closure
    //~| NOTE not a function or closure

    #[inline] impl S { }
    //~^ ERROR attribute should be applied to function or closure
    //~| NOTE not a function or closure
}

#[no_link]
//~^ ERROR attribute should be applied to an `extern crate` item
mod no_link {
    //~^ NOTE not an `extern crate` item

    mod inner { #![no_link] }
    //~^ ERROR attribute should be applied to an `extern crate` item
    //~| NOTE not an `extern crate` item

    #[no_link] fn f() { }
    //~^ ERROR attribute should be applied to an `extern crate` item
    //~| NOTE not an `extern crate` item

    #[no_link] struct S;
    //~^ ERROR attribute should be applied to an `extern crate` item
    //~| NOTE not an `extern crate` item

    #[no_link]type T = S;
    //~^ ERROR attribute should be applied to an `extern crate` item
    //~| NOTE not an `extern crate` item

    #[no_link] impl S { }
    //~^ ERROR attribute should be applied to an `extern crate` item
    //~| NOTE not an `extern crate` item
}

#[export_name = "2200"]
//~^ ERROR attribute should be applied to a function or static
mod export_name {
    //~^ NOTE not a function or static

    mod inner { #![export_name="2200"] }
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static

    #[export_name = "2200"] fn f() { }

    #[export_name = "2200"] struct S;
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static

    #[export_name = "2200"] type T = S;
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static

    #[export_name = "2200"] impl S { }
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static
}

#[main]
//~^ ERROR: `main` attribute can only be used on functions
mod main {
    mod inner { #![main] }
    //~^ ERROR: `main` attribute can only be used on functions

    // for `fn f()` case, see feature-gate-main.rs

    #[main] struct S;
    //~^ ERROR: `main` attribute can only be used on functions

    #[main] type T = S;
    //~^ ERROR: `main` attribute can only be used on functions

    #[main] impl S { }
    //~^ ERROR: `main` attribute can only be used on functions
}

#[start]
//~^ ERROR: `start` attribute can only be used on functions
mod start {
    mod inner { #![start] }
    //~^ ERROR: `start` attribute can only be used on functions

    // for `fn f()` case, see feature-gate-start.rs

    #[start] struct S;
    //~^ ERROR: `start` attribute can only be used on functions

    #[start] type T = S;
    //~^ ERROR: `start` attribute can only be used on functions

    #[start] impl S { }
    //~^ ERROR: `start` attribute can only be used on functions
}

fn main() {}
