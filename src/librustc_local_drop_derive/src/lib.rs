// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_use]
extern crate synstructure;
#[macro_use]
extern crate quote;
extern crate proc_macro2;

fn local_drop_derive(s: synstructure::Structure) -> proc_macro2::TokenStream {
    let body = s.each(|bi| quote!{
        // FIXME Crashes when printed: ::rustc_data_structures::local_drop
        ::rustc_data_structures::local_drop::check(#bi);
    });

    s.unsafe_bound_impl(quote!(::rustc_data_structures::local_drop::LocalDrop), quote!{
        fn check() {
            ::rustc_data_structures::local_drop::ensure_no_drop::<Self>();
            |this: &Self| {
                match *this { #body }
            };
        }
    })
}
decl_derive!([LocalDrop] => local_drop_derive);
