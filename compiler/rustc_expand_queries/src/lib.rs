#![allow(internal_features, reason = "proc macro internals")]
#![feature(proc_macro_internals)]

mod derive;

pub fn provide(providers: &mut rustc_middle::query::Providers) {
    providers.derive_macro_expansion = derive::derive_macro_expansion;
}
