use proc_macro::TokenStream;
use proc_macro2::Span;
use syn::{
    Token, Ident, Type, Attribute, ReturnType, Expr,
    braced, parenthesized, parse_macro_input,
};
use syn::parse::{Result, Parse, ParseStream};
use syn::punctuated::Punctuated;
use quote::quote;
use crate::tt::TS;

struct IdentOrWild(Ident);

impl Parse for IdentOrWild {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        Ok(if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            IdentOrWild(Ident::new("_", Span::call_site()))
        } else {
            IdentOrWild(input.parse()?)
        })
    }
}

enum QueryAttribute {
    Desc(Option<Ident>, Punctuated<Expr, Token![,]>),
    Cache(Option<Ident>, Expr),
    FatalCycle,
}

impl Parse for QueryAttribute {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let attr: Ident = input.parse()?;
        if attr == "desc" {
            let attr_content;
            braced!(attr_content in input);
            let tcx = if attr_content.peek(Token![|]) {
                attr_content.parse::<Token![|]>()?;
                let tcx = attr_content.parse()?;
                attr_content.parse::<Token![|]>()?;
                Some(tcx)
            } else {
                None
            };
            let desc = attr_content.parse_terminated(Expr::parse)?;
            if !attr_content.is_empty() {
                panic!("unexpected tokens in block");
            };
            Ok(QueryAttribute::Desc(tcx, desc))
        } else if attr == "cache" {
            let attr_content;
            braced!(attr_content in input);
            let tcx = if attr_content.peek(Token![|]) {
                attr_content.parse::<Token![|]>()?;
                let tcx = attr_content.parse()?;
                attr_content.parse::<Token![|]>()?;
                Some(tcx)
            } else {
                None
            };
            let expr = attr_content.parse()?;
            if !attr_content.is_empty() {
                panic!("unexpected tokens in block");
            };
            Ok(QueryAttribute::Cache(tcx, expr))
        } else if attr == "fatal_cycle" {
            Ok(QueryAttribute::FatalCycle)
        } else {
            panic!("unknown query modifier {}", attr)
        }
    }
}

struct Query {
    attrs: List<QueryAttribute>,
    name: Ident,
    key: IdentOrWild,
    arg: Type,
    result: ReturnType,
}

fn check_attributes(attrs: Vec<Attribute>) {
    for attr in attrs {
        let path = attr.path;
        let path = quote! { #path };
        let path = TS(&path);

        if path != TS(&quote! { doc }) {
            panic!("attribute `{}` not supported on queries", path.0)
        }
    }
}

impl Parse for Query {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        check_attributes(input.call(Attribute::parse_outer)?);

        let query: Ident = input.parse()?;
        if query != "query" {
            panic!("expected `query`");
        }
        let name: Ident = input.parse()?;
        let arg_content;
        parenthesized!(arg_content in input);
        let key = arg_content.parse()?;
        arg_content.parse::<Token![:]>()?;
        let arg = arg_content.parse()?;
        if !arg_content.is_empty() {
            panic!("expected only one query argument");
        };
        let result = input.parse()?;

        let content;
        braced!(content in input);
        let attrs = content.parse()?;

        Ok(Query {
            attrs,
            name,
            key,
            arg,
            result,
        })
    }
}

struct List<T>(Vec<T>);

impl<T: Parse> Parse for List<T> {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut list = Vec::new();
        while !input.is_empty() {
            list.push(input.parse()?);
        }
        Ok(List(list))
    }
}

struct Group {
    name: Ident,
    queries: List<Query>,
}

impl Parse for Group {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        Ok(Group {
            name,
            queries: content.parse()?,
        })
    }
}

fn camel_case(string: &str) -> String {
    let mut pos = vec![0];
    for (i, c) in string.chars().enumerate() {
        if c == '_' {
            pos.push(i + 1);
        }
    }
    string.chars().enumerate().filter(|c| c.1 != '_').flat_map(|(i, c)| {
        if pos.contains(&i) {
            c.to_uppercase().collect::<Vec<char>>()
        } else {
            vec![c]
        }
    }).collect()
}

pub fn rustc_queries(input: TokenStream) -> TokenStream {
    let groups = parse_macro_input!(input as List<Group>);

    let mut query_stream = quote! {};
    let mut query_description_stream = quote! {};
    let mut dep_node_def_stream = quote! {};
    let mut dep_node_force_stream = quote! {};

    for group in groups.0 {
        let mut group_stream = quote! {};
        for query in &group.queries.0 {
            let name = &query.name;
            let dep_node_name = Ident::new(
                &camel_case(&name.to_string()),
                name.span());
            let arg = &query.arg;
            let key = &query.key.0;
            let result_full = &query.result;
            let result = match query.result {
                ReturnType::Default => quote! { -> () },
                _ => quote! { #result_full },
            };

            // Find out if we should cache the query on disk
            let cache = query.attrs.0.iter().find_map(|attr| match attr {
                QueryAttribute::Cache(tcx, expr) => Some((tcx, expr)),
                _ => None,
            }).map(|(tcx, expr)| {
                let tcx = tcx.as_ref().map(|t| quote! { #t }).unwrap_or(quote! { _ });
                quote! {
                    #[inline]
                    fn cache_on_disk(#tcx: TyCtxt<'_, 'tcx, 'tcx>, #key: Self::Key) -> bool {
                        #expr
                    }

                    #[inline]
                    fn try_load_from_disk(
                        tcx: TyCtxt<'_, 'tcx, 'tcx>,
                        id: SerializedDepNodeIndex
                    ) -> Option<Self::Value> {
                        tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
                    }
                }
            });

            let fatal_cycle = query.attrs.0.iter().find_map(|attr| match attr {
                QueryAttribute::FatalCycle => Some(()),
                _ => None,
            }).map(|_| quote! { fatal_cycle }).unwrap_or(quote! {});

            group_stream.extend(quote! {
                [#fatal_cycle] fn #name: #dep_node_name(#arg) #result,
            });

            let desc = query.attrs.0.iter().find_map(|attr| match attr {
                QueryAttribute::Desc(tcx, desc) => Some((tcx, desc)),
                _ => None,
            }).map(|(tcx, desc)| {
                let tcx = tcx.as_ref().map(|t| quote! { #t }).unwrap_or(quote! { _ });
                quote! {
                    fn describe(
                        #tcx: TyCtxt<'_, '_, '_>,
                        #key: #arg,
                    ) -> Cow<'static, str> {
                        format!(#desc).into()
                    }
                }
            });

            if desc.is_some() || cache.is_some() {
                let cache = cache.unwrap_or(quote! {});
                let desc = desc.unwrap_or(quote! {});

                query_description_stream.extend(quote! {
                    impl<'tcx> QueryDescription<'tcx> for queries::#name<'tcx> {
                        #desc
                        #cache
                    }
                });
            }

            dep_node_def_stream.extend(quote! {
                [] #dep_node_name(#arg),
            });
            dep_node_force_stream.extend(quote! {
                DepKind::#dep_node_name => {
                    if let Some(key) = RecoverKey::recover($tcx, $dep_node) {
                        force_ex!($tcx, #name, key);
                    } else {
                        return false;
                    }
                }
            });
        }
        let name = &group.name;
        query_stream.extend(quote! {
            #name { #group_stream },
        });
    }
    TokenStream::from(quote! {
        macro_rules! rustc_query_append {
            ([$($macro:tt)*][$($other:tt)*]) => {
                $($macro)* {
                    $($other)*

                    #query_stream

                }
            }
        }
        macro_rules! rustc_dep_node_append {
            ([$($macro:tt)*][$($other:tt)*]) => {
                $($macro)*(
                    $($other)*

                    #dep_node_def_stream
                );
            }
        }
        macro_rules! rustc_dep_node_force {
            ([$dep_node:expr, $tcx:expr] $($other:tt)*) => {
                match $dep_node.kind {
                    $($other)*

                    #dep_node_force_stream
                }
            }
        }
        #query_description_stream
    })
}
