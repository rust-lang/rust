// At time of authorship, #[proc_macro_derive = "2500"] will emit an
// error when it occurs on a mod (apart from crate-level), but will
// not descend further into the mod for other occurrences of the same
// error.
//
// This file sits on its own because the "weird" occurrences here
// signal errors, making it incompatible with the "warnings only"
// nature of issue-43106-gating-of-builtin-attrs.rs

#[proc_macro_derive()]
//~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions
mod proc_macro_derive1 {
    mod inner { #![proc_macro_derive()] }
    // (no error issued here if there was one on outer module)
}

mod proc_macro_derive2 {
    mod inner { #![proc_macro_derive()] }
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions

    #[proc_macro_derive()] fn f() { }
    //~^ ERROR the `#[proc_macro_derive]` attribute is only usable with crates of the `proc-macro`

    #[proc_macro_derive()] struct S;
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions

    #[proc_macro_derive()] type T = S;
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions

    #[proc_macro_derive()] impl S { }
    //~^ ERROR the `#[proc_macro_derive]` attribute may only be used on bare functions
}

fn main() {}
