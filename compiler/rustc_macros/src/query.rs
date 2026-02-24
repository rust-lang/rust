use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    AttrStyle, Attribute, Block, Error, Expr, Ident, Pat, ReturnType, Token, Type, braced,
    parenthesized, parse_macro_input, token,
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

/// Declaration of a compiler query.
///
/// ```ignore (illustrative)
/// /// Doc comment for `my_query`.
/// //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^              doc_comments
/// query my_query(key: DefId) -> Value { anon }
/// //    ^^^^^^^^                               name
/// //             ^^^                           key_pat
/// //                  ^^^^^                    key_ty
/// //                         ^^^^^^^^          return_ty
/// //                                    ^^^^   modifiers
/// ```
struct Query {
    doc_comments: Vec<Attribute>,
    name: Ident,

    /// Parameter name for the key, or an arbitrary irrefutable pattern (e.g. `_`).
    key_pat: Pat,
    key_ty: Type,
    return_ty: ReturnType,

    modifiers: QueryModifiers,
}

impl Parse for Query {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut doc_comments = check_attributes(input.call(Attribute::parse_outer)?)?;

        // Parse the query declaration. Like `query type_of(key: DefId) -> Ty<'tcx>`
        input.parse::<kw::query>()?;
        let name: Ident = input.parse()?;

        // `(key: DefId)`
        let parens_content;
        parenthesized!(parens_content in input);
        let key_pat = Pat::parse_single(&parens_content)?;
        parens_content.parse::<Token![:]>()?;
        let key_ty = parens_content.parse::<Type>()?;
        let _trailing_comma = parens_content.parse::<Option<Token![,]>>()?;

        // `-> Value`
        let return_ty = input.parse::<ReturnType>()?;

        // Parse the query modifiers
        let braces_content;
        braced!(braces_content in input);
        let modifiers = parse_query_modifiers(&braces_content)?;

        // If there are no doc-comments, give at least some idea of what
        // it does by showing the query description.
        if doc_comments.is_empty() {
            doc_comments.push(doc_comment_from_desc(&modifiers.desc.expr_list)?);
        }

        Ok(Query { doc_comments, modifiers, name, key_pat, key_ty, return_ty })
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

struct Desc {
    modifier: Ident,
    expr_list: Punctuated<Expr, Token![,]>,
}

struct CacheOnDiskIf {
    modifier: Ident,
    block: Block,
}

struct QueryModifiers {
    /// The description of the query.
    desc: Desc,

    /// Use this type for the in-memory cache.
    arena_cache: Option<Ident>,

    /// Cache the query to disk if the `Block` returns true.
    cache_on_disk_if: Option<CacheOnDiskIf>,

    /// A cycle error for this query aborting the compilation with a fatal error.
    cycle_fatal: Option<Ident>,

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
    let mut cache_on_disk_if = None;
    let mut desc = None;
    let mut cycle_fatal = None;
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
            // `desc { "foo {}", tcx.item_path(key) }`
            let attr_content;
            braced!(attr_content in input);
            let expr_list = attr_content.parse_terminated(Expr::parse, Token![,])?;
            try_insert!(desc = Desc { modifier, expr_list });
        } else if modifier == "cache_on_disk_if" {
            // Parse a cache-on-disk modifier like:
            // `cache_on_disk_if { tcx.is_typeck_child(key.to_def_id()) }`
            let block = input.parse()?;
            try_insert!(cache_on_disk_if = CacheOnDiskIf { modifier, block });
        } else if modifier == "arena_cache" {
            try_insert!(arena_cache = modifier);
        } else if modifier == "cycle_fatal" {
            try_insert!(cycle_fatal = modifier);
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
        cache_on_disk_if,
        desc,
        cycle_fatal,
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

/// Contains token streams that are used to accumulate per-query helper
/// functions, to be used by the final output of `rustc_queries!`.
///
/// Helper items typically have the same name as the query they relate to,
/// and expect to be interpolated into a dedicated module.
#[derive(Default)]
struct HelperTokenStreams {
    description_fns_stream: proc_macro2::TokenStream,
    cache_on_disk_if_fns_stream: proc_macro2::TokenStream,
}

fn make_helpers_for_query(query: &Query, streams: &mut HelperTokenStreams) {
    let Query { name, key_pat, key_ty, modifiers, .. } = &query;

    // Replace span for `name` to make rust-analyzer ignore it.
    let mut erased_name = name.clone();
    erased_name.set_span(Span::call_site());

    // Generate a function to check whether we should cache the query to disk, for some key.
    if let Some(CacheOnDiskIf { block, .. }) = modifiers.cache_on_disk_if.as_ref() {
        // `pass_by_value`: some keys are marked with `rustc_pass_by_value`, but we take keys by
        // reference here.
        // FIXME: `pass_by_value` is badly named; `allow(rustc::pass_by_value)` actually means
        // "allow pass by reference of `rustc_pass_by_value` types".
        streams.cache_on_disk_if_fns_stream.extend(quote! {
            #[allow(unused_variables, rustc::pass_by_value)]
            #[inline]
            pub fn #erased_name<'tcx>(tcx: TyCtxt<'tcx>, #key_pat: &#key_ty) -> bool
            #block
        });
    }

    let Desc { expr_list, .. } = &modifiers.desc;

    let desc = quote! {
        #[allow(unused_variables)]
        pub fn #erased_name<'tcx>(tcx: TyCtxt<'tcx>, #key_pat: #key_ty) -> String {
            format!(#expr_list)
        }
    };

    streams.description_fns_stream.extend(quote! {
        #desc
    });
}

/// Add hints for rust-analyzer
fn add_to_analyzer_stream(query: &Query, analyzer_stream: &mut proc_macro2::TokenStream) {
    // Add links to relevant modifiers

    let modifiers = &query.modifiers;

    let mut modifiers_stream = quote! {};

    let name = &modifiers.desc.modifier;
    modifiers_stream.extend(quote! {
        crate::query::modifiers::#name;
    });

    if let Some(CacheOnDiskIf { modifier, .. }) = &modifiers.cache_on_disk_if {
        modifiers_stream.extend(quote! {
            crate::query::modifiers::#modifier;
        });
    }

    macro_rules! doc_link {
        ( $( $modifier:ident ),+ $(,)? ) => {
            $(
                if let Some(name) = &modifiers.$modifier {
                    modifiers_stream.extend(quote! {
                        crate::query::modifiers::#name;
                    });
                }
            )+
        }
    }

    doc_link!(
        arena_cache,
        cycle_fatal,
        cycle_delay_bug,
        cycle_stash,
        no_hash,
        anon,
        eval_always,
        depth_limit,
        separate_provide_extern,
        feedable,
        return_result_from_ensure_ok,
    );

    let name = &query.name;

    // Replace span for `name` to make rust-analyzer ignore it.
    let mut erased_name = name.clone();
    erased_name.set_span(Span::call_site());

    let result = &query.return_ty;

    // This dead code exists to instruct rust-analyzer about the link between the `rustc_queries`
    // query names and the corresponding produced provider. The issue is that by nature of this
    // macro producing a higher order macro that has all its token in the macro declaration we lose
    // any meaningful spans, resulting in rust-analyzer being unable to make the connection between
    // the query name and the corresponding providers field. The trick to fix this is to have
    // `rustc_queries` emit a field access with the given name's span which allows it to successfully
    // show references / go to definition to the corresponding provider assignment which is usually
    // the more interesting place.
    let ra_hint = quote! {
        let crate::query::Providers { #name: _, .. };
    };

    analyzer_stream.extend(quote! {
        #[inline(always)]
        fn #erased_name<'tcx>() #result {
            #ra_hint
            #modifiers_stream
            loop {}
        }
    });
}

pub(super) fn rustc_queries(input: TokenStream) -> TokenStream {
    let queries = parse_macro_input!(input as List<Query>);

    let mut query_stream = quote! {};
    let mut helpers = HelperTokenStreams::default();
    let mut analyzer_stream = quote! {};
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
        let Query { doc_comments, name, key_ty, return_ty, modifiers, .. } = &query;

        // Normalize an absent return type into `-> ()` to make macro-rules parsing easier.
        let return_ty = match return_ty {
            ReturnType::Default => quote! { -> () },
            ReturnType::Type(..) => quote! { #return_ty },
        };

        let mut modifiers_out = vec![];

        macro_rules! passthrough {
            ( $( $modifier:ident ),+ $(,)? ) => {
                $( if let Some($modifier) = &modifiers.$modifier {
                    modifiers_out.push(quote! { (#$modifier) });
                }; )+
            }
        }

        passthrough!(
            arena_cache,
            cycle_fatal,
            cycle_delay_bug,
            cycle_stash,
            no_hash,
            anon,
            eval_always,
            feedable,
            depth_limit,
            separate_provide_extern,
            return_result_from_ensure_ok,
        );

        // If there was a `cache_on_disk_if` modifier in the real input, pass
        // on a synthetic `(cache_on_disk)` modifier that can be inspected by
        // macro-rules macros.
        if modifiers.cache_on_disk_if.is_some() {
            modifiers_out.push(quote! { (cache_on_disk) });
        }

        // This uses the span of the query definition for the commas,
        // which can be important if we later encounter any ambiguity
        // errors with any of the numerous macro_rules! macros that
        // we use. Using the call-site span would result in a span pointing
        // at the entire `rustc_queries!` invocation, which wouldn't
        // be very useful.
        let span = name.span();
        let modifiers_stream = quote_spanned! { span => #(#modifiers_out),* };

        // Add the query to the group
        query_stream.extend(quote! {
            #(#doc_comments)*
            [#modifiers_stream]
            fn #name(#key_ty) #return_ty,
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
        }

        add_to_analyzer_stream(&query, &mut analyzer_stream);
        make_helpers_for_query(&query, &mut helpers);
    }

    let HelperTokenStreams { description_fns_stream, cache_on_disk_if_fns_stream } = helpers;

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

        // Add hints for rust-analyzer
        mod _analyzer_hints {
            use super::*;
            #analyzer_stream
        }

        /// Functions that format a human-readable description of each query
        /// and its key, as specified by the `desc` query modifier.
        ///
        /// (The leading `_` avoids collisions with actual query names when
        /// expanded in `rustc_middle::queries`, and makes this macro-generated
        /// module easier to search for.)
        pub mod _description_fns {
            use super::*;
            #description_fns_stream
        }

        // FIXME(Zalathar): Instead of declaring these functions directly, can
        // we put them in a macro and then expand that macro downstream in
        // `rustc_query_impl`, where the functions are actually used?
        pub mod _cache_on_disk_if_fns {
            use super::*;
            #cache_on_disk_if_fns_stream
        }

        #errors
    })
}
