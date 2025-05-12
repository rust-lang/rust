// A proc-macro in 2015 that has an RPIT without `use<>` that would cause a
// problem with 2024 capturing rules.

extern crate proc_macro;
use proc_macro::TokenStream;

#[proc_macro]
pub fn pm_rpit(input: TokenStream) -> TokenStream {
    "fn test_pm(x: &Vec<i32>) -> impl std::fmt::Display {
    x[0]
}

pub fn from_pm() {
    let mut x = vec![];
    x.push(1);

    let element = test_pm(&x);
    x.push(2);
    println!(\"{element}\");
}
"
    .parse()
    .unwrap()
}
