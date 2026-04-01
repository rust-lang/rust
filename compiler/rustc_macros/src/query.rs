use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{quote, quote_spanned};
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    AttrStyle, Attribute, Error, Expr, Ident, Pat, ReturnType, Token, Type, braced, parenthesized,
    parse_macro_input, token,
};

mod kw {
    syn::custom_keyword!(non_query);
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

/// Declaration of a non-query dep kind.
/// ```ignore (illustrative)
/// /// Doc comment for `MyNonQuery`.
/// //  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  doc_comments
/// non_query MyNonQuery
/// //        ^^^^^^^^^^               name
/// ```
struct NonQuery {
    doc_comments: Vec<Attribute>,
    name: Ident,
}

enum QueryEntry {
    Query(Query),
    NonQuery(NonQuery),
}

impl Parse for QueryEntry {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut doc_comments = check_attributes(input.call(Attribute::parse_outer)?)?;

        // Try the non-query case first.
        if input.parse::<kw::non_query>().is_ok() {
            let name: Ident = input.parse()?;
            return Ok(QueryEntry::NonQuery(NonQuery { doc_comments, name }));
        }

        // Parse the query declaration. Like `query type_of(key: DefId) -> Ty<'tcx>`
        if input.parse::<kw::query>().is_err() {
            return Err(input.error("expected `query` or `non_query`"));
        }
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

        Ok(QueryEntry::Query(Query { doc_comments, modifiers, name, key_pat, key_ty, return_ty }))
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
    // This ident is always `desc` but we need it for its span, for `crate::query::modifiers`.
    modifier: Ident,
    expr_list: Punctuated<Expr, Token![,]>,
}

/// See `rustc_middle::query::modifiers` for documentation of each query modifier.
struct QueryModifiers {
    // tidy-alphabetical-start
    arena_cache: Option<Ident>,
    cache_on_disk: Option<Ident>,
    depth_limit: Option<Ident>,
    desc: Desc,
    eval_always: Option<Ident>,
    feedable: Option<Ident>,
    handle_cycle_error: Option<Ident>,
    no_force: Option<Ident>,
    no_hash: Option<Ident>,
    separate_provide_extern: Option<Ident>,
    // tidy-alphabetical-end
}

fn parse_query_modifiers(input: ParseStream<'_>) -> Result<QueryModifiers> {
    // tidy-alphabetical-start
    let mut arena_cache = None;
    let mut cache_on_disk = None;
    let mut depth_limit = None;
    let mut desc = None;
    let mut eval_always = None;
    let mut feedable = None;
    let mut handle_cycle_error = None;
    let mut no_force = None;
    let mut no_hash = None;
    let mut separate_provide_extern = None;
    // tidy-alphabetical-end

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

        if modifier == "arena_cache" {
            try_insert!(arena_cache = modifier);
        } else if modifier == "cache_on_disk" {
            try_insert!(cache_on_disk = modifier);
        } else if modifier == "depth_limit" {
            try_insert!(depth_limit = modifier);
        } else if modifier == "desc" {
            // Parse a description modifier like:
            // `desc { "foo {}", tcx.item_path(key) }`
            let attr_content;
            braced!(attr_content in input);
            let expr_list = attr_content.parse_terminated(Expr::parse, Token![,])?;
            try_insert!(desc = Desc { modifier, expr_list });
        } else if modifier == "eval_always" {
            try_insert!(eval_always = modifier);
        } else if modifier == "feedable" {
            try_insert!(feedable = modifier);
        } else if modifier == "handle_cycle_error" {
            try_insert!(handle_cycle_error = modifier);
        } else if modifier == "no_force" {
            try_insert!(no_force = modifier);
        } else if modifier == "no_hash" {
            try_insert!(no_hash = modifier);
        } else if modifier == "separate_provide_extern" {
            try_insert!(separate_provide_extern = modifier);
        } else {
            return Err(Error::new(modifier.span(), "unknown query modifier"));
        }
    }
    let Some(desc) = desc else {
        return Err(input.error("no description provided"));
    };
    Ok(QueryModifiers {
        // tidy-alphabetical-start
        arena_cache,
        cache_on_disk,
        depth_limit,
        desc,
        eval_always,
        feedable,
        handle_cycle_error,
        no_force,
        no_hash,
        separate_provide_extern,
        // tidy-alphabetical-end
    })
}

// Does `ret_ty` match `Result<_, ErrorGuaranteed>`?
fn returns_error_guaranteed(ret_ty: &ReturnType) -> bool {
    use ::syn::*;
    if let ReturnType::Type(_, ret_ty) = ret_ty
        && let Type::Path(type_path) = ret_ty.as_ref()
        && let Some(PathSegment { ident, arguments }) = type_path.path.segments.last()
        && ident == "Result"
        && let PathArguments::AngleBracketed(args) = arguments
        && args.args.len() == 2
        && let GenericArgument::Type(ty) = &args.args[1]
        && let Type::Path(type_path) = ty
        && type_path.path.is_ident("ErrorGuaranteed")
    {
        true
    } else {
        false
    }
}

fn make_modifiers_stream(query: &Query) -> proc_macro2::TokenStream {
    let QueryModifiers {
        // tidy-alphabetical-start
        arena_cache,
        cache_on_disk,
        depth_limit,
        desc,
        eval_always,
        feedable,
        handle_cycle_error,
        no_force,
        no_hash,
        separate_provide_extern,
        // tidy-alphabetical-end
    } = &query.modifiers;

    // tidy-alphabetical-start
    let arena_cache = arena_cache.is_some();
    let cache_on_disk = cache_on_disk.is_some();
    let depth_limit = depth_limit.is_some();
    let desc = {
        // Put a description closure in the `desc` modifier.
        let key_pat = &query.key_pat;
        let key_ty = &query.key_ty;
        let desc_expr_list = &desc.expr_list;
        quote! {
            {
                #[allow(unused_variables)]
                |tcx: TyCtxt<'tcx>, #key_pat: #key_ty| format!(#desc_expr_list)
            }
        }
    };
    let eval_always = eval_always.is_some();
    let feedable = feedable.is_some();
    let handle_cycle_error = handle_cycle_error.is_some();
    let no_force = no_force.is_some();
    let no_hash = no_hash.is_some();
    let returns_error_guaranteed = returns_error_guaranteed(&query.return_ty);
    let separate_provide_extern = separate_provide_extern.is_some();
    // tidy-alphabetical-end

    // Giving an input span to the modifier names in the modifier list seems
    // to give slightly more helpful errors when one of the callback macros
    // fails to parse the modifier list.
    let query_name_span = query.name.span();
    quote_spanned! {
        query_name_span =>
        // Search for (QMODLIST) to find all occurrences of this query modifier list.
        // tidy-alphabetical-start
        arena_cache: #arena_cache,
        cache_on_disk: #cache_on_disk,
        depth_limit: #depth_limit,
        desc: #desc,
        eval_always: #eval_always,
        feedable: #feedable,
        handle_cycle_error: #handle_cycle_error,
        no_force: #no_force,
        no_hash: #no_hash,
        returns_error_guaranteed: #returns_error_guaranteed,
        separate_provide_extern: #separate_provide_extern,
        // tidy-alphabetical-end
    }
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

/// Add hints for rust-analyzer
fn add_to_analyzer_stream(query: &Query, analyzer_stream: &mut proc_macro2::TokenStream) {
    // Add links to relevant modifiers

    let modifiers = &query.modifiers;

    let mut modifiers_stream = quote! {};

    let name = &modifiers.desc.modifier;
    modifiers_stream.extend(quote! {
        crate::query::modifiers::#name;
    });

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
        // tidy-alphabetical-start
        arena_cache,
        cache_on_disk,
        depth_limit,
        // `desc` is handled above
        eval_always,
        feedable,
        handle_cycle_error,
        no_force,
        no_hash,
        separate_provide_extern,
        // tidy-alphabetical-end
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
    // `rustc_queries` emit a field access with the given name's span which allows it to
    // successfully show references / go to definition to the corresponding provider assignment
    // which is usually the more interesting place.
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
    let queries = parse_macro_input!(input as List<QueryEntry>);

    let mut query_stream = quote! {};
    let mut non_query_stream = quote! {};
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
        let query = match query {
            QueryEntry::Query(query) => query,
            QueryEntry::NonQuery(NonQuery { doc_comments, name }) => {
                // Get the exceptional non-query case out of the way first.
                non_query_stream.extend(quote! {
                    #(#doc_comments)*
                    #name,
                });
                continue;
            }
        };

        let Query { doc_comments, name, key_ty, return_ty, modifiers, .. } = &query;

        // Normalize an absent return type into `-> ()` to make macro-rules parsing easier.
        let return_ty = match return_ty {
            ReturnType::Default => quote! { -> () },
            ReturnType::Type(..) => quote! { #return_ty },
        };

        let modifiers_stream = make_modifiers_stream(&query);

        // Add the query to the group
        query_stream.extend(quote! {
            #(#doc_comments)*
            fn #name(#key_ty) #return_ty
            { #modifiers_stream }
        });

        if let Some(feedable) = &modifiers.feedable {
            assert!(
                modifiers.eval_always.is_none(),
                feedable.span(),
                "Query {name} cannot be both `feedable` and `eval_always`."
            );
        }

        add_to_analyzer_stream(&query, &mut analyzer_stream);
    }

    TokenStream::from(quote! {
        /// Higher-order macro that invokes the specified macro with (a) a list of all query
        /// signatures (including modifiers), and (b) a list of non-query names. This allows
        /// multiple simpler macros to each have access to these lists.
        #[rustc_macro_transparency = "semiopaque"] // Use `macro_rules!` hygiene.
        pub macro rustc_with_all_queries {
            (
                // The macro to invoke once, on all queries and non-queries.
                $macro:ident!
            ) => {
                $macro! {
                    queries { #query_stream }
                    non_queries { #non_query_stream }
                }
            }
        }

        // Add hints for rust-analyzer
        mod _analyzer_hints {
            use super::*;
            #analyzer_stream
        }

        #errors
    })
}
