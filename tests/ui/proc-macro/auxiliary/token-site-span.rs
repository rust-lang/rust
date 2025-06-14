// Testing token span hygiene.

//@ proc-macro: mixed-site-span.rs

extern crate mixed_site_span;

use mixed_site_span::declare_macro;

pub struct TokenItem;

#[macro_export]
macro_rules! invoke_with_crate {
    ($s:ident $i:ident) => { with_crate!{$crate $s $i} };
}

#[macro_export]
macro_rules! invoke_with_ident {
    ($s:ident $i:ident) => { with_crate!{krate $s $i} };
    ($m:ident $s:ident $i:ident) => { with_crate!{$m $s $i} };
}

macro_rules! local {() => {
    declare_macro!{$crate input use_input_crate}
    declare_macro!{$crate mixed use_mixed_crate}
    declare_macro!{$crate call use_call_crate}
}}
local!{}
declare_macro!{krate input use_input_krate}
declare_macro!{krate mixed use_mixed_krate}
declare_macro!{krate call use_call_krate}
