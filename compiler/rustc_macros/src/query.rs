use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    braced, parenthesized, parse_macro_input, parse_quote, token, AttrStyle, Attribute, Block,
    Error, Expr, Ident, Pat, ReturnType, Token, Type,
};

mod kw {
    syn::custom_keyword!(query);
}

/// Ensures only doc comment attributes are used
fn check_attributes(attrs: Vec<Attribute>) -> Result<Vec<Attribute>> {
    let inner = |attr: Attribute| {
        if !attr.path.is_ident("doc") {
            Err(Error::new(attr.span(), "attributes not supported on queries"))
        } else if attr.style != AttrStyle::Outer {
            Err(Error::new(
                attr.span(),
                "attributes must be outer attributes (`///`), not inner attributes",
            ))
        } else {
            Ok(attr)
        }
    };
    attrs.into_iter().map(inner).collect()
}

/// A compiler query. `query ... { ... }`
struct Query {
    doc_comments: Vec<Attribute>,
    modifiers: QueryModifiers,
    name: Ident,
    key: Pat,
    arg: Type,
    result: ReturnType,
}

impl Parse for Query {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut doc_comments = check_attributes(input.call(Attribute::parse_outer)?)?;

        // Parse the query declaration. Like `query type_of(key: DefId) -> Ty<'tcx>`
        input.parse::<kw::query>()?;
        let name: Ident = input.parse()?;
        let arg_content;
        parenthesized!(arg_content in input);
        let key = arg_content.parse()?;
        arg_content.parse::<Token![:]>()?;
        let arg = arg_content.parse()?;
        let result = input.parse()?;

        // Parse the query modifiers
        let content;
        braced!(content in input);
        let modifiers = parse_query_modifiers(&content)?;

        // If there are no doc-comments, give at least some idea of what
        // it does by showing the query description.
        if doc_comments.is_empty() {
            doc_comments.push(doc_comment_from_desc(&modifiers.desc.1)?);
        }

        Ok(Query { doc_comments, modifiers, name, key, arg, result })
    }
}

/// A type used to greedily parse another type until the input is empty.
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

struct QueryModifiers {
    /// The description of the query.
    desc: (Option<Ident>, Punctuated<Expr, Token![,]>),

    /// Use this type for the in-memory cache.
    arena_cache: Option<Ident>,

    /// Cache the query to disk if the `Block` returns true.
    cache: Option<(Option<Pat>, Block)>,

    /// A cycle error for this query aborting the compilation with a fatal error.
    fatal_cycle: Option<Ident>,

    /// A cycle error results in a delay_bug call
    cycle_delay_bug: Option<Ident>,

    /// Don't hash the result, instead just mark a query red if it runs
    no_hash: Option<Ident>,

    /// Generate a dep node based on the dependencies of the query
    anon: Option<Ident>,

    /// Always evaluate the query, ignoring its dependencies
    eval_always: Option<Ident>,

    /// Whether the query has a call depth limit
    depth_limit: Option<Ident>,

    /// Use a separate query provider for local and extern crates
    separate_provide_extern: Option<Ident>,

    /// Always remap the ParamEnv's constness before hashing.
    remap_env_constness: Option<Ident>,
}

fn parse_query_modifiers(input: ParseStream<'_>) -> Result<QueryModifiers> {
    let mut arena_cache = None;
    let mut cache = None;
    let mut desc = None;
    let mut fatal_cycle = None;
    let mut cycle_delay_bug = None;
    let mut no_hash = None;
    let mut anon = None;
    let mut eval_always = None;
    let mut depth_limit = None;
    let mut separate_provide_extern = None;
    let mut remap_env_constness = None;

    while !input.is_empty() {
        let modifier: Ident = input.parse()?;

        macro_rules! try_insert {
            ($name:ident = $expr:expr) => {
                if $name.is_some() {
                    return Err(Error::new(modifier.span(), "duplicate modifier"));
                }
                $name = Some($expr);
            };
        }

        if modifier == "desc" {
            // Parse a description modifier like:
            // `desc { |tcx| "foo {}", tcx.item_path(key) }`
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
            let list = attr_content.parse_terminated(Expr::parse)?;
            try_insert!(desc = (tcx, list));
        } else if modifier == "cache_on_disk_if" {
            // Parse a cache modifier like:
            // `cache(tcx) { |tcx| key.is_local() }`
            let args = if input.peek(token::Paren) {
                let args;
                parenthesized!(args in input);
                let tcx = args.parse()?;
                Some(tcx)
            } else {
                None
            };
            let block = input.parse()?;
            try_insert!(cache = (args, block));
        } else if modifier == "arena_cache" {
            try_insert!(arena_cache = modifier);
        } else if modifier == "fatal_cycle" {
            try_insert!(fatal_cycle = modifier);
        } else if modifier == "cycle_delay_bug" {
            try_insert!(cycle_delay_bug = modifier);
        } else if modifier == "no_hash" {
            try_insert!(no_hash = modifier);
        } else if modifier == "anon" {
            try_insert!(anon = modifier);
        } else if modifier == "eval_always" {
            try_insert!(eval_always = modifier);
        } else if modifier == "depth_limit" {
            try_insert!(depth_limit = modifier);
        } else if modifier == "separate_provide_extern" {
            try_insert!(separate_provide_extern = modifier);
        } else if modifier == "remap_env_constness" {
            try_insert!(remap_env_constness = modifier);
        } else {
            return Err(Error::new(modifier.span(), "unknown query modifier"));
        }
    }
    let Some(desc) = desc else {
        return Err(input.error("no description provided"));
    };
    Ok(QueryModifiers {
        arena_cache,
        cache,
        desc,
        fatal_cycle,
        cycle_delay_bug,
        no_hash,
        anon,
        eval_always,
        depth_limit,
        separate_provide_extern,
        remap_env_constness,
    })
}

fn doc_comment_from_desc(list: &Punctuated<Expr, token::Comma>) -> Result<Attribute> {
    use ::syn::*;
    let mut iter = list.iter();
    let format_str: String = match iter.next() {
        Some(&Expr::Lit(ExprLit { lit: Lit::Str(ref lit_str), .. })) => {
            lit_str.value().replace("`{}`", "{}") // We add them later anyways for consistency
        }
        _ => return Err(Error::new(list.span(), "Expected a string literal")),
    };
    let mut fmt_fragments = format_str.split("{}");
    let mut doc_string = fmt_fragments.next().unwrap().to_string();
    iter.map(::quote::ToTokens::to_token_stream).zip(fmt_fragments).for_each(
        |(tts, next_fmt_fragment)| {
            use ::core::fmt::Write;
            write!(
                &mut doc_string,
                " `{}` {}",
                tts.to_string().replace(" . ", "."),
                next_fmt_fragment,
            )
            .unwrap();
        },
    );
    let doc_string = format!("[query description - consider adding a doc-comment!] {}", doc_string);
    Ok(parse_quote! { #[doc = #doc_string] })
}

/// Add the impl of QueryDescription for the query to `impls` if one is requested
fn add_query_description_impl(query: &Query, impls: &mut proc_macro2::TokenStream) {
    let name = &query.name;
    let key = &query.key;
    let modifiers = &query.modifiers;

    // Find out if we should cache the query on disk
    let cache = if let Some((args, expr)) = modifiers.cache.as_ref() {
        let tcx = args.as_ref().map(|t| quote! { #t }).unwrap_or_else(|| quote! { _ });
        // expr is a `Block`, meaning that `{ #expr }` gets expanded
        // to `{ { stmts... } }`, which triggers the `unused_braces` lint.
        quote! {
            #[allow(unused_variables, unused_braces)]
            #[inline]
            fn cache_on_disk(#tcx: TyCtxt<'tcx>, #key: &Self::Key) -> bool {
                #expr
            }
        }
    } else {
        quote! {
            #[inline]
            fn cache_on_disk(_: TyCtxt<'tcx>, _: &Self::Key) -> bool {
                false
            }
        }
    };

    let (tcx, desc) = &modifiers.desc;
    let tcx = tcx.as_ref().map_or_else(|| quote! { _ }, |t| quote! { #t });

    let desc = quote! {
        #[allow(unused_variables)]
        fn describe(tcx: QueryCtxt<'tcx>, key: Self::Key) -> String {
            let (#tcx, #key) = (*tcx, key);
            ::rustc_middle::ty::print::with_no_trimmed_paths!(
                format!(#desc)
            )
        }
    };

    impls.extend(quote! {
        (#name) => {
            #desc
            #cache
        };
    });
}

pub fn rustc_queries(input: TokenStream) -> TokenStream {
    let queries = parse_macro_input!(input as List<Query>);

    let mut query_stream = quote! {};
    let mut query_description_stream = quote! {};
    let mut dep_node_def_stream = quote! {};
    let mut cached_queries = quote! {};

    for query in queries.0 {
        let Query { name, arg, modifiers, .. } = &query;
        let result_full = &query.result;
        let result = match query.result {
            ReturnType::Default => quote! { -> () },
            _ => quote! { #result_full },
        };

        if modifiers.cache.is_some() {
            cached_queries.extend(quote! {
                #name,
            });
        }

        let mut attributes = Vec::new();

        macro_rules! passthrough {
            ( $( $modifier:ident ),+ $(,)? ) => {
                $( if let Some($modifier) = &modifiers.$modifier {
                    attributes.push(quote! { (#$modifier) });
                }; )+
            }
        }

        passthrough!(
            fatal_cycle,
            arena_cache,
            cycle_delay_bug,
            no_hash,
            anon,
            eval_always,
            depth_limit,
            separate_provide_extern,
            remap_env_constness,
        );

        if modifiers.cache.is_some() {
            attributes.push(quote! { (cache) });
        }

        // This uses the span of the query definition for the commas,
        // which can be important if we later encounter any ambiguity
        // errors with any of the numerous macro_rules! macros that
        // we use. Using the call-site span would result in a span pointing
        // at the entire `rustc_queries!` invocation, which wouldn't
        // be very useful.
        let span = name.span();
        let attribute_stream = quote_spanned! {span=> #(#attributes),*};
        let doc_comments = query.doc_comments.iter();
        // Add the query to the group
        query_stream.extend(quote! {
            #(#doc_comments)*
            [#attribute_stream] fn #name(#arg) #result,
        });

        // Create a dep node for the query
        dep_node_def_stream.extend(quote! {
            [#attribute_stream] #name(#arg),
        });

        add_query_description_impl(&query, &mut query_description_stream);
    }

    TokenStream::from(quote! {
        #[macro_export]
        macro_rules! rustc_query_append {
            ($macro:ident !) => {
                $macro! {
                    #query_stream
                }
            }
        }
        macro_rules! rustc_dep_node_append {
            ($macro:ident! [$($other:tt)*]) => {
                $macro!(
                    $($other)*

                    #dep_node_def_stream
                );
            }
        }
        #[macro_export]
        macro_rules! rustc_cached_queries {
            ( $macro:ident! ) => {
                $macro!(#cached_queries);
            }
        }
        #[macro_export]
        macro_rules! rustc_query_description {
            #query_description_stream
        }
    })
}
