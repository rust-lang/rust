use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    AttrStyle, Attribute, Block, Error, Expr, Ident, Pat, ReturnType, Token, Type, braced,
    parenthesized, parse_macro_input, parse_quote, token,
};

mod kw {
    syn::custom_keyword!(query);
}

/// Ensures only doc comment attributes are used
fn check_attributes(attrs: Vec<Attribute>) -> Result<Vec<Attribute>> {
    let inner = |attr: Attribute| {
        if !attr.path().is_ident("doc") {
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
        let key = Pat::parse_single(&arg_content)?;
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

    /// A cycle error results in a stashed cycle error that can be unstashed and canceled later
    cycle_stash: Option<Ident>,

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

    /// Generate a `feed` method to set the query's value from another query.
    feedable: Option<Ident>,

    /// When this query is called via `tcx.ensure_ok()`, it returns
    /// `Result<(), ErrorGuaranteed>` instead of `()`. If the query needs to
    /// be executed, and that execution returns an error, the error result is
    /// returned to the caller.
    ///
    /// If execution is skipped, a synthetic `Ok(())` is returned, on the
    /// assumption that a query with all-green inputs must have succeeded.
    ///
    /// Can only be applied to queries with a return value of
    /// `Result<_, ErrorGuaranteed>`.
    return_result_from_ensure_ok: Option<Ident>,
}

fn parse_query_modifiers(input: ParseStream<'_>) -> Result<QueryModifiers> {
    let mut arena_cache = None;
    let mut cache = None;
    let mut desc = None;
    let mut fatal_cycle = None;
    let mut cycle_delay_bug = None;
    let mut cycle_stash = None;
    let mut no_hash = None;
    let mut anon = None;
    let mut eval_always = None;
    let mut depth_limit = None;
    let mut separate_provide_extern = None;
    let mut feedable = None;
    let mut return_result_from_ensure_ok = None;

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
            let list = attr_content.parse_terminated(Expr::parse, Token![,])?;
            try_insert!(desc = (tcx, list));
        } else if modifier == "cache_on_disk_if" {
            // Parse a cache modifier like:
            // `cache(tcx) { |tcx| key.is_local() }`
            let args = if input.peek(token::Paren) {
                let args;
                parenthesized!(args in input);
                let tcx = Pat::parse_single(&args)?;
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
        } else if modifier == "cycle_stash" {
            try_insert!(cycle_stash = modifier);
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
        } else if modifier == "feedable" {
            try_insert!(feedable = modifier);
        } else if modifier == "return_result_from_ensure_ok" {
            try_insert!(return_result_from_ensure_ok = modifier);
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
        cycle_stash,
        no_hash,
        anon,
        eval_always,
        depth_limit,
        separate_provide_extern,
        feedable,
        return_result_from_ensure_ok,
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
    let doc_string = format!("[query description - consider adding a doc-comment!] {doc_string}");
    Ok(parse_quote! { #[doc = #doc_string] })
}

/// Add the impl of QueryDescription for the query to `impls` if one is requested
fn add_query_desc_cached_impl(
    query: &Query,
    descs: &mut proc_macro2::TokenStream,
    cached: &mut proc_macro2::TokenStream,
) {
    let Query { name, key, modifiers, .. } = &query;

    // This dead code exists to instruct rust-analyzer about the link between the `rustc_queries`
    // query names and the corresponding produced provider. The issue is that by nature of this
    // macro producing a higher order macro that has all its token in the macro declaration we lose
    // any meaningful spans, resulting in rust-analyzer being unable to make the connection between
    // the query name and the corresponding providers field. The trick to fix this is to have
    // `rustc_queries` emit a field access with the given name's span which allows it to succesfully
    // show references / go to definition to the correspondig provider assignment which is usually
    // the more interesting place.
    let ra_hint = quote! {
        let crate::query::Providers { #name: _, .. };
    };

    // Find out if we should cache the query on disk
    let cache = if let Some((args, expr)) = modifiers.cache.as_ref() {
        let tcx = args.as_ref().map(|t| quote! { #t }).unwrap_or_else(|| quote! { _ });
        // expr is a `Block`, meaning that `{ #expr }` gets expanded
        // to `{ { stmts... } }`, which triggers the `unused_braces` lint.
        // we're taking `key` by reference, but some rustc types usually prefer being passed by value
        quote! {
            #[allow(unused_variables, unused_braces, rustc::pass_by_value)]
            #[inline]
            pub fn #name<'tcx>(#tcx: TyCtxt<'tcx>, #key: &crate::query::queries::#name::Key<'tcx>) -> bool {
                #ra_hint
                #expr
            }
        }
    } else {
        quote! {
            // we're taking `key` by reference, but some rustc types usually prefer being passed by value
            #[allow(rustc::pass_by_value)]
            #[inline]
            pub fn #name<'tcx>(_: TyCtxt<'tcx>, _: &crate::query::queries::#name::Key<'tcx>) -> bool {
                #ra_hint
                false
            }
        }
    };

    let (tcx, desc) = &modifiers.desc;
    let tcx = tcx.as_ref().map_or_else(|| quote! { _ }, |t| quote! { #t });

    let desc = quote! {
        #[allow(unused_variables)]
        pub fn #name<'tcx>(tcx: TyCtxt<'tcx>, key: crate::query::queries::#name::Key<'tcx>) -> String {
            let (#tcx, #key) = (tcx, key);
            ::rustc_middle::ty::print::with_no_trimmed_paths!(
                format!(#desc)
            )
        }
    };

    descs.extend(quote! {
        #desc
    });

    cached.extend(quote! {
        #cache
    });
}

pub(super) fn rustc_queries(input: TokenStream) -> TokenStream {
    let queries = parse_macro_input!(input as List<Query>);

    let mut query_stream = quote! {};
    let mut query_description_stream = quote! {};
    let mut query_cached_stream = quote! {};
    let mut feedable_queries = quote! {};
    let mut errors = quote! {};

    macro_rules! assert {
        ( $cond:expr, $span:expr, $( $tt:tt )+ ) => {
            if !$cond {
                errors.extend(
                    Error::new($span, format!($($tt)+)).into_compile_error(),
                );
            }
        }
    }

    for query in queries.0 {
        let Query { name, arg, modifiers, .. } = &query;
        let result_full = &query.result;
        let result = match query.result {
            ReturnType::Default => quote! { -> () },
            _ => quote! { #result_full },
        };

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
            cycle_stash,
            no_hash,
            anon,
            eval_always,
            depth_limit,
            separate_provide_extern,
            return_result_from_ensure_ok,
        );

        if modifiers.cache.is_some() {
            attributes.push(quote! { (cache) });
        }
        // Pass on the cache modifier
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
        let doc_comments = &query.doc_comments;
        // Add the query to the group
        query_stream.extend(quote! {
            #(#doc_comments)*
            [#attribute_stream] fn #name(#arg) #result,
        });

        if let Some(feedable) = &modifiers.feedable {
            assert!(
                modifiers.anon.is_none(),
                feedable.span(),
                "Query {name} cannot be both `feedable` and `anon`."
            );
            assert!(
                modifiers.eval_always.is_none(),
                feedable.span(),
                "Query {name} cannot be both `feedable` and `eval_always`."
            );
            feedable_queries.extend(quote! {
                #(#doc_comments)*
                [#attribute_stream] fn #name(#arg) #result,
            });
        }

        add_query_desc_cached_impl(&query, &mut query_description_stream, &mut query_cached_stream);
    }

    TokenStream::from(quote! {
        /// Higher-order macro that invokes the specified macro with a prepared
        /// list of all query signatures (including modifiers).
        ///
        /// This allows multiple simpler macros to each have access to the list
        /// of queries.
        #[macro_export]
        macro_rules! rustc_with_all_queries {
            (
                // The macro to invoke once, on all queries (plus extras).
                $macro:ident!

                // Within [], an optional list of extra "query" signatures to
                // pass to the given macro, in addition to the actual queries.
                $( [$($extra_fake_queries:tt)*] )?
            ) => {
                $macro! {
                    $( $($extra_fake_queries)* )?
                    #query_stream
                }
            }
        }
        macro_rules! rustc_feedable_queries {
            ( $macro:ident! ) => {
                $macro!(#feedable_queries);
            }
        }
        pub mod descs {
            use super::*;
            #query_description_stream
        }
        pub mod cached {
            use super::*;
            #query_cached_stream
        }
        #errors
    })
}
