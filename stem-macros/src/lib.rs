extern crate proc_macro;

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, FnArg, ItemFn, ReturnType, Type};

#[proc_macro_attribute]
pub fn main(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let fn_name = &func.sig.ident;

    if func.sig.asyncness.is_some() {
        return syn::Error::new_spanned(
            &func.sig.ident,
            "#[stem::main] does not support async functions",
        )
        .to_compile_error()
        .into();
    }

    // Ensure return type is !
    let is_never = matches!(
        func.sig.output,
        ReturnType::Type(_, ref ty) if matches!(**ty, Type::Never(_))
    );
    if !is_never {
        return syn::Error::new_spanned(&func.sig.output, "#[stem::main] functions must return !")
            .to_compile_error()
            .into();
    }

    // Accept either no args or a single usize argument.
    let (accepts_arg, arg_error) = match func.sig.inputs.len() {
        0 => (false, None),
        1 => match func.sig.inputs.first().unwrap() {
            FnArg::Typed(pat_ty) => {
                if let Type::Path(path) = pat_ty.ty.as_ref() {
                    if path.path.is_ident("usize") {
                        (true, None)
                    } else {
                        (false, Some("expected argument type usize"))
                    }
                } else {
                    (false, Some("expected argument type usize"))
                }
            }
            _ => (false, Some("unsupported argument pattern")),
        },
        _ => (false, Some("expected zero or one argument")),
    };

    if let Some(msg) = arg_error {
        return syn::Error::new_spanned(&func.sig.inputs, msg)
            .to_compile_error()
            .into();
    }

    let call_user = if accepts_arg {
        quote! { #fn_name(__stem_arg) }
    } else {
        quote! { #fn_name() }
    };

    let expanded = quote! {
        #func

        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn stem_user_main(__stem_arg: usize) -> ! {
            #call_user
        }
    };

    expanded.into()
}
