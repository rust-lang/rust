use proc_macro::TokenStream;
use proc_macro_error::proc_macro_error;
use quote::quote;

mod gen;
mod parser;

#[proc_macro_attribute]
#[proc_macro_error]
pub fn autodiff(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut params = parser::parse(args.into(), input.clone().into());
    let (primal, adjoint) = (gen::primal_fnc(&mut params), gen::adjoint_fnc(&params));

    let res = quote!(
        #primal
        #adjoint
    );

    res.into()
}

#[test]
pub fn expanding() {
    macrotest::expand("tests/expand/*.rs");
}

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
