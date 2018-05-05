// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic

#![feature(proc_macro, proc_macro_non_items)]
#![crate_type = "proc-macro"]

extern crate proc_macro;

use proc_macro::{TokenStream, TokenTree, Spacing, Literal, quote};

#[proc_macro]
pub fn count_compound_ops(input: TokenStream) -> TokenStream {
    assert_eq!(count_compound_ops_helper(quote!(++ (&&) 4@a)), 3);
    let l = Literal::u32_suffixed(count_compound_ops_helper(input));
    TokenTree::from(l).into()
}

fn count_compound_ops_helper(input: TokenStream) -> u32 {
    let mut count = 0;
    for token in input {
        match &token {
            TokenTree::Punct(tt) if tt.spacing() == Spacing::Alone => {
                count += 1;
            }
            TokenTree::Group(tt) => {
                count += count_compound_ops_helper(tt.stream());
            }
            _ => {}
        }
    }
    count
}
