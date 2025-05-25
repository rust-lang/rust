//~ NOTE: not an `extern crate` item
//~^ NOTE: not a free function, impl method or static
//~^^ NOTE: not a function or closure
// This is testing whether various builtin attributes signals an
// error or warning when put in "weird" places.
//
// (This file sits on its own because it actually signals an error,
// which would mess up the treatment of other cases in
// issue-43106-gating-of-builtin-attrs.rs)


#![macro_export]
//~^ ERROR: `macro_export` attribute cannot be used at crate level
#![rustc_main]
//~^ ERROR: `rustc_main` attribute cannot be used at crate level
//~| ERROR: use of an internal attribute [E0658]
//~| NOTE: the `#[rustc_main]` attribute is an internal implementation detail that will never be stable
//~| NOTE: the `#[rustc_main]` attribute is used internally to specify test entry point function
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
//~^ ERROR: attribute should be applied to a free function, impl method or static
#![inline]
//~^ ERROR: attribute should be applied to function or closure
#[inline]
//~^ ERROR attribute should be applied to function or closure
mod inline {
    //~^ NOTE not a function or closure
    //~| NOTE the inner attribute doesn't annotate this module
    //~| NOTE the inner attribute doesn't annotate this module
    //~| NOTE the inner attribute doesn't annotate this module
    //~| NOTE the inner attribute doesn't annotate this module
    //~| NOTE the inner attribute doesn't annotate this module

    mod inner { #![inline] }
    //~^ ERROR attribute should be applied to function or closure
    //~| NOTE not a function or closure

    #[inline = "2100"] fn f() { }
    //~^ ERROR valid forms for the attribute are
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
//~^ ERROR attribute should be applied to a free function, impl method or static
mod export_name {
    //~^ NOTE not a free function, impl method or static

    mod inner { #![export_name="2200"] }
    //~^ ERROR attribute should be applied to a free function, impl method or static
    //~| NOTE not a free function, impl method or static

    #[export_name = "2200"] fn f() { }

    #[export_name = "2200"] struct S;
    //~^ ERROR attribute should be applied to a free function, impl method or static
    //~| NOTE not a free function, impl method or static

    #[export_name = "2200"] type T = S;
    //~^ ERROR attribute should be applied to a free function, impl method or static
    //~| NOTE not a free function, impl method or static

    #[export_name = "2200"] impl S { }
    //~^ ERROR attribute should be applied to a free function, impl method or static
    //~| NOTE not a free function, impl method or static

    trait Tr {
        #[export_name = "2200"] fn foo();
        //~^ ERROR attribute should be applied to a free function, impl method or static
        //~| NOTE not a free function, impl method or static

        #[export_name = "2200"] fn bar() {}
        //~^ ERROR attribute should be applied to a free function, impl method or static
        //~| NOTE not a free function, impl method or static
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
