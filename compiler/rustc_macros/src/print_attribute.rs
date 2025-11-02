use proc_macro2::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;
use syn::{Data, Fields, Ident};
use synstructure::Structure;

fn print_fields(name: &Ident, fields: &Fields) -> (TokenStream, TokenStream) {
    let string_name = name.to_string();
    let mut disps = vec![quote! {let mut __printed_anything = false;}];

    match fields {
        Fields::Named(fields_named) => {
            let mut field_names = Vec::new();

            for field in &fields_named.named {
                let name = field.ident.as_ref().unwrap();
                let string_name = name.to_string();
                disps.push(quote! {
                    if #name.should_render() {
                        if __printed_anything {
                            __p.word_space(",");
                        }
                        __p.word(#string_name);
                        __p.word(":");
                        __p.nbsp();
                        __printed_anything = true;
                    }
                    #name.print_attribute(__p);
                });
                field_names.push(name);
            }

            (
                quote! { {#(#field_names),*} },
                quote! {
                    __p.word(#string_name);
                    if true #(&& !#field_names.should_render())* {
                        return;
                    }

                    __p.nbsp();
                    __p.word("{");
                    #(#disps)*
                    __p.word("}");
                },
            )
        }
        Fields::Unnamed(fields_unnamed) => {
            let mut field_names = Vec::new();

            for idx in 0..fields_unnamed.unnamed.len() {
                let name = format_ident!("f{idx}");
                disps.push(quote! {
                    if #name.should_render() {
                        if __printed_anything {
                            __p.word_space(",");
                        }
                        __printed_anything = true;
                    }
                    #name.print_attribute(__p);
                });
                field_names.push(name);
            }

            (
                quote! { (#(#field_names),*) },
                quote! {
                    __p.word(#string_name);

                    if true #(&& !#field_names.should_render())* {
                        return;
                    }

                    __p.popen();
                    #(#disps)*
                    __p.pclose();
                },
            )
        }
        Fields::Unit => (quote! {}, quote! { __p.word(#string_name) }),
    }
}

pub(crate) fn print_attribute(input: Structure<'_>) -> TokenStream {
    let span_error = |span, message: &str| {
        quote_spanned! { span => const _: () = ::core::compile_error!(#message); }
    };

    // Must be applied to an enum type.
    let code = match &input.ast().data {
        Data::Enum(e) => {
            let arms = e
                .variants
                .iter()
                .map(|x| {
                    let ident = &x.ident;
                    let (pat, code) = print_fields(ident, &x.fields);

                    quote! {
                        Self::#ident #pat => {#code}
                    }
                })
                .collect::<Vec<_>>();

            quote! {
                match self {
                    #(#arms)*
                }
            }
        }
        Data::Struct(s) => {
            let (pat, code) = print_fields(&input.ast().ident, &s.fields);
            quote! {
                let Self #pat = self;
                #code
            }
        }
        Data::Union(u) => {
            return span_error(u.union_token.span(), "can't derive PrintAttribute on unions");
        }
    };

    #[allow(keyword_idents_2024)]
    input.gen_impl(quote! {
        #[allow(unused)]
        gen impl PrintAttribute for @Self {
            fn should_render(&self) -> bool { true }
            fn print_attribute(&self, __p: &mut rustc_ast_pretty::pp::Printer) { #code }
        }
    })
}
