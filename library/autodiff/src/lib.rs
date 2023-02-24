use proc_macro::TokenStream;
use proc_macro_error::proc_macro_error;
use quote::quote;

mod parser;
//mod gen;

#[proc_macro_attribute]
#[proc_macro_error]
pub fn autodiff(args: TokenStream, input: TokenStream) -> TokenStream {
    let mut params = parser::parse(args.into(), input.clone().into());
    //let (body, fnc_source) = gen::generate_body(input.into(), &params);
    //let header = gen::generate_header(&params);

    //// generate function

    //let out = if params.block.is_none() {
    //    let sig = &params.sig;
    //    quote!(
    //        #[autodiff_into]
    //        #fnc_source

    //        #header
    //        #sig {
    //            #body
    //        }
    //    )
    //} else {
    //    params.sig.ident = params.header.name.get_ident().unwrap().clone();
    //    let sig = &params.sig;

    //    quote!(
    //        #[autodiff_into]
    //        #fnc_source

    //        #header
    //        #sig {
    //            #body
    //        }
    //    )
    //};

    //out.into()
    quote!().into()
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
