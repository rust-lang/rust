//@ edition: 2024
//@ aux-build: macro-source.rs
//@ aux-build: macro-2018.rs
//@ aux-build: macro-2024.rs
//@ check-pass

#![feature(prelude_import)]

extern crate macro_2018;
extern crate macro_2024;
extern crate macro_source;

#[prelude_import]
use macro_source::trait_prelude::*;

fn main() {
    // Ordinary names in the prelude are resolved using the identifier's
    // edition. Exercise both namespaces of the redirected unit struct.
    let _: macro_2018::redirected_type!() = macro_source::trait_prelude::OldItem;
    let _: macro_2024::redirected_type!() =
        macro_source::trait_prelude::CurrentItem;
    let _: macro_source::trait_prelude::OldItem = macro_2018::redirected_value!();
    let _: macro_source::trait_prelude::CurrentItem =
        macro_2024::redirected_value!();

    // Both calls search the same external prelude module. Trait discovery must
    // select the redirect using each macro-generated method name's edition
    // rather than reuse the first cached result.
    let _: OldMarker = macro_2018::call_redirected_trait!();
    let _: CurrentMarker = macro_2024::call_redirected_trait!();
}
