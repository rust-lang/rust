#![allow(unused)]

// compile-flags: -Zunstable-options
// run-pass

extern crate core;

//1. Removed library module
use core::removed_lib_features::*;      //~ ERROR use of removed library feature `removed_lib_features` [E0658]
use core::removed_macro;                //~ ERROR use of removed library feature `removed_macro_item` [E0658]

fn main() {
    // 2. Removed library macro
    removed_macro!();                   //~ ERROR use of removed library feature `removed_macro_item` [E0658]

    // 3. Removed library trait
    // This triggers *both* removed_trait_impl and removed_trait_fn
    Impl::removed_fn();                 //~ ERROR use of removed library feature `removed_trait_impl` [E0658]
                                        //~| ERROR use of removed library feature `removed_trait_fn` [E0658]

    // Same here for const
    let _ = Impl::REMOVED_TRAIT_CONST;  //~ ERROR use of removed library feature `removed_trait_impl` [E0658]
                                        //~| ERROR use of removed library feature `removed_trait_fn` [E0658]

    // 4. Removed library const
    let _ = const_removed::use_removed_const(); //~ ERROR use of removed library feature `removed_const_item` [E0658]


    // 5. Missing Firlds
    removed_no_reason(); //~ ERROR use of removed library feature `removed_no_reason` [E0658]
}
