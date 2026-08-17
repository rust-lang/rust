#![allow(internal_features, reason = "proc macro internals")]
#![feature(proc_macro_internals)]

mod derive;

pub fn enable() {
    rustc_expand::proc_macro::EXPAND_DERIVE_CACHED.swap(&(derive::expand_derive_macro as _));
}

pub fn provide(providers: &mut rustc_middle::query::Providers) {
    providers.derive_macro_expansion = derive::derive_macro_expansion;
}
