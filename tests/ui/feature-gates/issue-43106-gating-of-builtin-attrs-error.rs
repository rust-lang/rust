//~ NOTE: not an `extern crate` item
// This is testing whether various builtin attributes signals an
// error or warning when put in "weird" places.
//
// (This file sits on its own because it actually signals an error,
// which would mess up the treatment of other cases in
// issue-43106-gating-of-builtin-attrs.rs)


#![macro_export]
//~^ ERROR:  `#[macro_export]` attribute cannot be used on crates
#![rustc_main]
//~^ ERROR: `#[rustc_main]` attribute cannot be used on crates
//~| ERROR: use of an internal attribute [E0658]
//~| NOTE: the `#[rustc_main]` attribute is an internal implementation detail that will never be stable
//~| NOTE: the `#[rustc_main]` attribute is used internally to specify test entry point function
#![repr()]
//~^ ERROR: `repr` attribute cannot be used at crate level
//~| WARN unused attribute
//~| NOTE empty list has no effect
#![path = "3800"]
//~^ ERROR: attribute cannot be used on
#![automatically_derived]
//~^ ERROR: attribute cannot be used on
#![no_mangle]
//~^ WARN may not be used in combination with `#[export_name]`
//~| NOTE is ignored
//~| NOTE requested on the command line
//~| WARN cannot be used on crates
//~| WARN previously accepted
#![no_link]
//~^ ERROR: attribute should be applied to an `extern crate` item
#![export_name = "2200"]
//~^ ERROR: attribute cannot be used on
//~| NOTE takes precedence
#![inline]
//~^ ERROR: attribute cannot be used on
#[inline]
//~^ ERROR attribute cannot be used on
mod inline {
    //~^ NOTE the inner attribute doesn't annotate this module

    mod inner { #![inline] }
    //~^ ERROR attribute cannot be used on

    #[inline = "2100"] fn f() { }
    //~^ ERROR valid forms for the attribute are
    //~| WARN this was previously accepted
    //~| NOTE `#[deny(ill_formed_attribute_input)]` (part of `#[deny(future_incompatible)]`) on by default
    //~| NOTE for more information, see issue #57571 <https://github.com/rust-lang/rust/issues/57571>

    #[inline] struct S;
    //~^ ERROR attribute cannot be used on

    #[inline] type T = S;
    //~^ ERROR attribute cannot be used on

    #[inline] impl S { }
    //~^ ERROR attribute cannot be used on
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
//~^ ERROR attribute cannot be used on
mod export_name {
    mod inner { #![export_name="2200"] }
    //~^ ERROR attribute cannot be used on

    #[export_name = "2200"] fn f() { }

    #[export_name = "2200"] struct S;
    //~^ ERROR attribute cannot be used on

    #[export_name = "2200"] type T = S;
    //~^ ERROR attribute cannot be used on

    #[export_name = "2200"] impl S { }
    //~^ ERROR attribute cannot be used on

    trait Tr {
        #[export_name = "2200"] fn foo();
        //~^ ERROR attribute cannot be used on

        #[export_name = "2200"] fn bar() {}
    }
}

#[repr(C)]
//~^ ERROR: attribute should be applied to a struct, enum, or union
mod repr {
//~^ NOTE not a struct, enum, or union
    mod inner { #![repr(C)] }
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union

    #[repr(C)] fn f() { }
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union

    struct S;

    #[repr(C)] type T = S;
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union

    #[repr(C)] impl S { }
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union
}


#[repr(Rust)]
//~^ ERROR: attribute should be applied to a struct, enum, or union
mod repr_rust {
//~^ NOTE not a struct, enum, or union
    mod inner { #![repr(Rust)] }
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union

    #[repr(Rust)] fn f() { }
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union

    struct S;

    #[repr(Rust)] type T = S;
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union

    #[repr(Rust)] impl S { }
    //~^ ERROR: attribute should be applied to a struct, enum, or union
    //~| NOTE not a struct, enum, or union
}

fn main() {}
