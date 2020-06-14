// This is testing whether various builtin attributes signals an
// error or warning when put in "weird" places.
//
// (This file sits on its own because it actually signals an error,
// which would mess up the treatment of other cases in
// issue-43106-gating-of-builtin-attrs.rs)

// ignore-tidy-linelength

// Crate-level is accepted, though it is almost certainly unused?
#![inline]

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

#[cold]
//~^ ERROR attribute should be applied to a function
mod cold {
    //~^ NOTE not a function

    mod inner { #![cold] }
    //~^ ERROR attribute should be applied to a function
    //~| NOTE not a function

    #[cold] fn f() { }

    #[cold] struct S;
    //~^ ERROR attribute should be applied to a function
    //~| NOTE not a function

    #[cold] type T = S;
    //~^ ERROR attribute should be applied to a function
    //~| NOTE not a function

    #[cold] impl S { }
    //~^ ERROR attribute should be applied to a function
    //~| NOTE not a function
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

#[link_name = "1900"]
//~^ ERROR attribute should be applied to a foreign function or static
mod link_name {
    //~^ NOTE not a foreign function or static

    #[link_name = "1900"]
    //~^ ERROR attribute should be applied to a foreign function or static
    //~| HELP try `#[link(name = "1900")]` instead
    extern { }
    //~^ NOTE not a foreign function or static

    mod inner { #![link_name="1900"] }
    //~^ ERROR attribute should be applied to a foreign function or static
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] fn f() { }
    //~^ ERROR attribute should be applied to a foreign function or static
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] struct S;
    //~^ ERROR attribute should be applied to a foreign function or static
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] type T = S;
    //~^ ERROR attribute should be applied to a foreign function or static
    //~| NOTE not a foreign function or static

    #[link_name = "1900"] impl S { }
    //~^ ERROR attribute should be applied to a foreign function or static
    //~| NOTE not a foreign function or static
}

#[link_section = "1800"]
//~^ ERROR attribute should be applied to a function or static
mod link_section {
    //~^ NOTE not a function or static

    mod inner { #![link_section="1800"] }
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static

    #[link_section = "1800"] fn f() { }

    #[link_section = "1800"] struct S;
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static

    #[link_section = "1800"] type T = S;
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static

    #[link_section = "1800"] impl S { }
    //~^ ERROR attribute should be applied to a function or static
    //~| NOTE not a function or static
}

fn main() {}
