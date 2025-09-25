// This file sits on its own because the occurrences here
// signal errors, making it incompatible with the "warnings only"
// nature of issue-43106-gating-of-builtin-attrs.rs

#[proc_macro_derive(Test)]
//~^ ERROR attribute cannot be used on
mod proc_macro_derive1 {
    mod inner { #![proc_macro_derive(Test)] }
    //~^ ERROR attribute cannot be used on
}

mod proc_macro_derive2 {
    mod inner { #![proc_macro_derive(Test)] }
    //~^ ERROR attribute cannot be used on

    #[proc_macro_derive(Test)] fn f() { }
    //~^ ERROR the `#[proc_macro_derive]` attribute is only usable with crates of the `proc-macro`

    #[proc_macro_derive(Test)] struct S;
    //~^ ERROR attribute cannot be used on

    #[proc_macro_derive(Test)] type T = S;
    //~^ ERROR attribute cannot be used on

    #[proc_macro_derive(Test)] impl S { }
    //~^ ERROR attribute cannot be used on
}

fn main() {}
