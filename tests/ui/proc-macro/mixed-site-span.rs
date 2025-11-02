// Proc macros using `mixed_site` spans exhibit usual properties of `macro_rules` hygiene.

//@ aux-build: token-site-span.rs
//@ proc-macro: mixed-site-span.rs
//@ ignore-backends: gcc

extern crate mixed_site_span;
extern crate token_site_span;

use mixed_site_span::{proc_macro_rules, with_crate};
use token_site_span::{
    invoke_with_crate, invoke_with_ident,
    use_input_crate, use_mixed_crate, use_call_crate,
    use_input_krate, use_mixed_krate, use_call_krate,
};

pub struct ItemUse;

fn main() {
    'label_use: loop {
        let local_use = 1;
        proc_macro_rules!();
        //~^ ERROR cannot find type `ItemUse` in crate `$crate`
        //~| ERROR use of undeclared label `'label_use`
        //~| ERROR cannot find value `local_use` in this scope
        ItemDef; // OK
        local_def; //~ ERROR cannot find value `local_def` in this scope
    }
}

// Successful resolutions of `mixed_site_span::proc_macro_item`
const _: () = {
    invoke_with_crate!{mixed proc_macro_item}
    invoke_with_ident!{mixed proc_macro_item}
    invoke_with_ident!{krate mixed proc_macro_item}
    with_crate!{krate mixed proc_macro_item}

    macro_rules! test {() => {
        invoke_with_ident!{$crate mixed proc_macro_item}
        with_crate!{$crate mixed proc_macro_item}
    }}
    test!();
};

// Failed resolutions of `proc_macro_item`
const _: () = {
    // token_site_span::proc_macro_item
    invoke_with_crate!{input proc_macro_item}            //~ ERROR unresolved import `$crate`
    invoke_with_ident!{input proc_macro_item}            //~ ERROR unresolved import `$crate`
    invoke_with_crate!{call proc_macro_item}             //~ ERROR unresolved import `$crate`
    invoke_with_ident!{call proc_macro_item}             //~ ERROR unresolved import `$crate`
    invoke_with_ident!{hello call proc_macro_item}       //~ ERROR unresolved import `$crate`

    // crate::proc_macro_item
    invoke_with_ident!{krate input proc_macro_item}      //~ ERROR unresolved import `$crate::proc_macro_item`
    with_crate!{krate input proc_macro_item}             //~ ERROR unresolved import `$crate::proc_macro_item`
    with_crate!{krate call proc_macro_item}              //~ ERROR unresolved import `$crate`

    macro_rules! test {() => {
        // crate::proc_macro_item
        invoke_with_ident!{$crate input proc_macro_item} //~ ERROR unresolved import `$crate`
        with_crate!{$crate input proc_macro_item}        //~ ERROR unresolved import `$crate`
        with_crate!{$crate call proc_macro_item}         //~ ERROR unresolved import `$crate`

        // token_site_span::proc_macro_item
        invoke_with_ident!{$crate call proc_macro_item}  //~ ERROR unresolved import `$crate`
    }}
    test!();
};

// Successful resolutions of `token_site_span::TokenItem`
const _: () = {
    invoke_with_crate!{input TokenItem}
    invoke_with_ident!{input TokenItem}
    invoke_with_crate!{call TokenItem}
    invoke_with_ident!{call TokenItem}
    invoke_with_ident!{hello call TokenItem}

    macro_rules! test {() => {
        invoke_with_ident!{$crate call TokenItem}
    }}
    test!();
};

// Failed resolutions of `TokenItem`
const _: () = {
    // crate::TokenItem
    invoke_with_ident!{krate input TokenItem}       //~ ERROR unresolved import `$crate::TokenItem`
    with_crate!{krate input TokenItem}              //~ ERROR unresolved import `$crate::TokenItem`
    with_crate!{krate call TokenItem}               //~ ERROR unresolved import `$crate`

    // mixed_site_span::TokenItem
    invoke_with_crate!{mixed TokenItem}             //~ ERROR unresolved import `$crate`
    invoke_with_ident!{mixed TokenItem}             //~ ERROR unresolved import `$crate`
    invoke_with_ident!{krate mixed TokenItem}       //~ ERROR unresolved import `$crate`
    with_crate!{krate mixed TokenItem}              //~ ERROR unresolved import `$crate`

    macro_rules! test {() => {
        // crate::TokenItem
        invoke_with_ident!{$crate input TokenItem}  //~ ERROR unresolved import `$crate`
        with_crate!{$crate input TokenItem}         //~ ERROR unresolved import `$crate`
        with_crate!{$crate call TokenItem}          //~ ERROR unresolved import `$crate`

        // mixed_site_span::TokenItem
        invoke_with_ident!{$crate mixed TokenItem}  //~ ERROR unresolved import `$crate`
        with_crate!{$crate mixed TokenItem}         //~ ERROR unresolved import `$crate`

    }}
    test!();
};


// Successful resolutions of `crate::ItemUse`
const _: () = {
    invoke_with_ident!{krate input ItemUse}
    with_crate!{krate input ItemUse}
    with_crate!{krate call ItemUse}

    macro_rules! test {() => {
        invoke_with_ident!{$crate input ItemUse}
        with_crate!{$crate input ItemUse}
        with_crate!{$crate call ItemUse}
    }}
    test!();
};

// Failed resolutions of `ItemUse`
const _: () = {
    // token_site_span::ItemUse
    invoke_with_crate!{input ItemUse}            //~ ERROR unresolved import `$crate`
    invoke_with_ident!{input ItemUse}            //~ ERROR unresolved import `$crate`

    // mixed_site_span::ItemUse
    invoke_with_crate!{mixed ItemUse}            //~ ERROR unresolved import `$crate`
    invoke_with_ident!{mixed ItemUse}            //~ ERROR unresolved import `$crate`
    invoke_with_ident!{krate mixed ItemUse}      //~ ERROR unresolved import `$crate`
    with_crate!{krate mixed ItemUse}             //~ ERROR unresolved import `$crate`

    invoke_with_crate!{call ItemUse}             //~ ERROR unresolved import `$crate`
    invoke_with_ident!{call ItemUse}             //~ ERROR unresolved import `$crate`
    invoke_with_ident!{hello call ItemUse}       //~ ERROR unresolved import `$crate`

    macro_rules! test {() => {
        invoke_with_ident!{$crate mixed ItemUse} //~ ERROR unresolved import `$crate`
        with_crate!{$crate mixed ItemUse}        //~ ERROR unresolved import `$crate`

        invoke_with_ident!{$crate call ItemUse}  //~ ERROR unresolved import `$crate`
    }}
    test!();
};


// Only mixed should see mixed_site_span::proc_macro_item
use_input_crate!{proc_macro_item}   //~ ERROR unresolved import `$crate`
use_input_krate!{proc_macro_item}   //~ ERROR unresolved import `$crate`
use_mixed_crate!{proc_macro_item}
use_mixed_krate!{proc_macro_item}
use_call_crate!{proc_macro_item}    //~ ERROR unresolved import `$crate`
use_call_krate!{proc_macro_item}    //~ ERROR unresolved import `$crate`

// Only mixed should fail to see token_site_span::TokenItem
use_input_crate!{TokenItem}
use_input_krate!{TokenItem}
use_mixed_crate!{TokenItem}         //~ ERROR unresolved import `$crate`
use_mixed_krate!{TokenItem}         //~ ERROR unresolved import `$crate`
use_call_crate!{TokenItem}
use_call_krate!{TokenItem}

// Everything should fail to see crate::ItemUse
use_input_crate!{ItemUse}           //~ ERROR unresolved import `$crate`
use_input_krate!{ItemUse}           //~ ERROR unresolved import `$crate`
use_mixed_crate!{ItemUse}           //~ ERROR unresolved import `$crate`
use_mixed_krate!{ItemUse}           //~ ERROR unresolved import `$crate`
use_call_crate!{ItemUse}            //~ ERROR unresolved import `$crate`
use_call_krate!{ItemUse}            //~ ERROR unresolved import `$crate`
