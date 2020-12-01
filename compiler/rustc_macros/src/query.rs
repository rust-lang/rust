use proc_macro::TokenStream;
use proc_macro2::{Delimiter, TokenTree};
use quote::quote;
use syn::parse::{Parse, ParseStream, Result};
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{
    braced, parenthesized, parse_macro_input, AttrStyle, Attribute, Block, Error, Expr, Ident,
    ReturnType, Token, Type,
};

mod kw {
    syn::custom_keyword!(query);
}

/// Ident or a wildcard `_`.
struct IdentOrWild(Ident);

impl Parse for IdentOrWild {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        Ok(if input.peek(Token![_]) {
            let underscore = input.parse::<Token![_]>()?;
            IdentOrWild(Ident::new("_", underscore.span()))
        } else {
            IdentOrWild(input.parse()?)
        })
    }
}

/// A modifier for a query
enum QueryModifier {
    /// The description of the query.
    Desc(Option<Ident>, Punctuated<Expr, Token![,]>),

    /// Use this type for the in-memory cache.
    Storage(Type),

    /// Cache the query to disk if the `Expr` returns true.
    Cache(Option<(IdentOrWild, IdentOrWild)>, Block),

    /// Custom code to load the query from disk.
    LoadCached(Ident, Ident, Block),

    /// A cycle error for this query aborting the compilation with a fatal error.
    FatalCycle,

    /// A cycle error results in a delay_bug call
    CycleDelayBug,

    /// Don't hash the result, instead just mark a query red if it runs
    NoHash,

    /// Generate a dep node based on the dependencies of the query
    Anon,

    /// Always evaluate the query, ignoring its dependencies
    EvalAlways,
}

impl Parse for QueryModifier {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let modifier: Ident = input.parse()?;
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
            let desc = attr_content.parse_terminated(Expr::parse)?;
            Ok(QueryModifier::Desc(tcx, desc))
        } else if modifier == "cache_on_disk_if" {
            // Parse a cache modifier like:
            // `cache(tcx, value) { |tcx| key.is_local() }`
            let has_args = if let TokenTree::Group(group) = input.fork().parse()? {
                group.delimiter() == Delimiter::Parenthesis
            } else {
                false
            };
            let args = if has_args {
                let args;
                parenthesized!(args in input);
                let tcx = args.parse()?;
                args.parse::<Token![,]>()?;
                let value = args.parse()?;
                Some((tcx, value))
            } else {
                None
            };
            let block = input.parse()?;
            Ok(QueryModifier::Cache(args, block))
        } else if modifier == "load_cached" {
            // Parse a load_cached modifier like:
            // `load_cached(tcx, id) { tcx.queries.on_disk_cache.try_load_query_result(tcx, id) }`
            let args;
            parenthesized!(args in input);
            let tcx = args.parse()?;
            args.parse::<Token![,]>()?;
            let id = args.parse()?;
            let block = input.parse()?;
            Ok(QueryModifier::LoadCached(tcx, id, block))
        } else if modifier == "storage" {
            let args;
            parenthesized!(args in input);
            let ty = args.parse()?;
            Ok(QueryModifier::Storage(ty))
        } else if modifier == "fatal_cycle" {
            Ok(QueryModifier::FatalCycle)
        } else if modifier == "cycle_delay_bug" {
            Ok(QueryModifier::CycleDelayBug)
        } else if modifier == "no_hash" {
            Ok(QueryModifier::NoHash)
        } else if modifier == "anon" {
            Ok(QueryModifier::Anon)
        } else if modifier == "eval_always" {
            Ok(QueryModifier::EvalAlways)
        } else {
            Err(Error::new(modifier.span(), "unknown query modifier"))
        }
    }
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
    modifiers: List<QueryModifier>,
    name: Ident,
    key: IdentOrWild,
    arg: Type,
    result: ReturnType,
}

impl Parse for Query {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let doc_comments = check_attributes(input.call(Attribute::parse_outer)?)?;

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
        let modifiers = content.parse()?;

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

/// A named group containing queries.
///
/// For now, the name is not used any more, but the capability remains interesting for future
/// developments of the query system.
struct Group {
    #[allow(unused)]
    name: Ident,
    queries: List<Query>,
}

impl Parse for Group {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let name: Ident = input.parse()?;
        let content;
        braced!(content in input);
        Ok(Group { name, queries: content.parse()? })
    }
}

struct QueryModifiers {
    /// The description of the query.
    desc: (Option<Ident>, Punctuated<Expr, Token![,]>),

    /// Use this type for the in-memory cache.
    storage: Option<Type>,

    /// Cache the query to disk if the `Block` returns true.
    cache: Option<(Option<(IdentOrWild, IdentOrWild)>, Block)>,

    /// Custom code to load the query from disk.
    load_cached: Option<(Ident, Ident, Block)>,

    /// A cycle error for this query aborting the compilation with a fatal error.
    fatal_cycle: bool,

    /// A cycle error results in a delay_bug call
    cycle_delay_bug: bool,

    /// Don't hash the result, instead just mark a query red if it runs
    no_hash: bool,

    /// Generate a dep node based on the dependencies of the query
    anon: bool,

    // Always evaluate the query, ignoring its dependencies
    eval_always: bool,
}

/// Process query modifiers into a struct, erroring on duplicates
fn process_modifiers(query: &mut Query) -> QueryModifiers {
    let mut load_cached = None;
    let mut storage = None;
    let mut cache = None;
    let mut desc = None;
    let mut fatal_cycle = false;
    let mut cycle_delay_bug = false;
    let mut no_hash = false;
    let mut anon = false;
    let mut eval_always = false;
    for modifier in query.modifiers.0.drain(..) {
        match modifier {
            QueryModifier::LoadCached(tcx, id, block) => {
                if load_cached.is_some() {
                    panic!("duplicate modifier `load_cached` for query `{}`", query.name);
                }
                load_cached = Some((tcx, id, block));
            }
            QueryModifier::Storage(ty) => {
                if storage.is_some() {
                    panic!("duplicate modifier `storage` for query `{}`", query.name);
                }
                storage = Some(ty);
            }
            QueryModifier::Cache(args, expr) => {
                if cache.is_some() {
                    panic!("duplicate modifier `cache` for query `{}`", query.name);
                }
                cache = Some((args, expr));
            }
            QueryModifier::Desc(tcx, list) => {
                if desc.is_some() {
                    panic!("duplicate modifier `desc` for query `{}`", query.name);
                }
                desc = Some((tcx, list));
            }
            QueryModifier::FatalCycle => {
                if fatal_cycle {
                    panic!("duplicate modifier `fatal_cycle` for query `{}`", query.name);
                }
                fatal_cycle = true;
            }
            QueryModifier::CycleDelayBug => {
                if cycle_delay_bug {
                    panic!("duplicate modifier `cycle_delay_bug` for query `{}`", query.name);
                }
                cycle_delay_bug = true;
            }
            QueryModifier::NoHash => {
                if no_hash {
                    panic!("duplicate modifier `no_hash` for query `{}`", query.name);
                }
                no_hash = true;
            }
            QueryModifier::Anon => {
                if anon {
                    panic!("duplicate modifier `anon` for query `{}`", query.name);
                }
                anon = true;
            }
            QueryModifier::EvalAlways => {
                if eval_always {
                    panic!("duplicate modifier `eval_always` for query `{}`", query.name);
                }
                eval_always = true;
            }
        }
    }
    let desc = desc.unwrap_or_else(|| {
        panic!("no description provided for query `{}`", query.name);
    });
    QueryModifiers {
        load_cached,
        storage,
        cache,
        desc,
        fatal_cycle,
        cycle_delay_bug,
        no_hash,
        anon,
        eval_always,
    }
}

/// Add the impl of QueryDescription for the query to `impls` if one is requested
fn add_query_description_impl(
    query: &Query,
    modifiers: QueryModifiers,
    impls: &mut proc_macro2::TokenStream,
) {
    let name = &query.name;
    let arg = &query.arg;
    let key = &query.key.0;

    // Find out if we should cache the query on disk
    let cache = if let Some((args, expr)) = modifiers.cache.as_ref() {
        let try_load_from_disk = if let Some((tcx, id, block)) = modifiers.load_cached.as_ref() {
            // Use custom code to load the query from disk
            quote! {
                #[inline]
                fn try_load_from_disk(
                    #tcx: TyCtxt<'tcx>,
                    #id: SerializedDepNodeIndex
                ) -> Option<Self::Value> {
                    #block
                }
            }
        } else {
            // Use the default code to load the query from disk
            quote! {
                #[inline]
                fn try_load_from_disk(
                    tcx: TyCtxt<'tcx>,
                    id: SerializedDepNodeIndex
                ) -> Option<Self::Value> {
                    tcx.queries.on_disk_cache.as_ref().and_then(|c| c.try_load_query_result(tcx, id))
                }
            }
        };

        let tcx = args
            .as_ref()
            .map(|t| {
                let t = &(t.0).0;
                quote! { #t }
            })
            .unwrap_or(quote! { _ });
        let value = args
            .as_ref()
            .map(|t| {
                let t = &(t.1).0;
                quote! { #t }
            })
            .unwrap_or(quote! { _ });
        // expr is a `Block`, meaning that `{ #expr }` gets expanded
        // to `{ { stmts... } }`, which triggers the `unused_braces` lint.
        quote! {
            #[inline]
            #[allow(unused_variables, unused_braces)]
            fn cache_on_disk(
                #tcx: TyCtxt<'tcx>,
                #key: &Self::Key,
                #value: Option<&Self::Value>
            ) -> bool {
                #expr
            }

            #try_load_from_disk
        }
    } else {
        if modifiers.load_cached.is_some() {
            panic!("load_cached modifier on query `{}` without a cache modifier", name);
        }
        quote! {}
    };

    let (tcx, desc) = modifiers.desc;
    let tcx = tcx.as_ref().map(|t| quote! { #t }).unwrap_or(quote! { _ });

    let desc = quote! {
        #[allow(unused_variables)]
        fn describe(
            #tcx: TyCtxt<'tcx>,
            #key: #arg,
        ) -> Cow<'static, str> {
            ::rustc_middle::ty::print::with_no_trimmed_paths(|| format!(#desc).into())
        }
    };

    impls.extend(quote! {
        impl<'tcx> QueryDescription<TyCtxt<'tcx>> for queries::#name<'tcx> {
            #desc
            #cache
        }
    });
}

pub fn rustc_queries(input: TokenStream) -> TokenStream {
    let groups = parse_macro_input!(input as List<Group>);

    let mut query_stream = quote! {};
    let mut query_description_stream = quote! {};
    let mut dep_node_def_stream = quote! {};
    let mut cached_queries = quote! {};

    for group in groups.0 {
        for mut query in group.queries.0 {
            let modifiers = process_modifiers(&mut query);
            let name = &query.name;
            let arg = &query.arg;
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

            // Pass on the fatal_cycle modifier
            if modifiers.fatal_cycle {
                attributes.push(quote! { fatal_cycle });
            };
            // Pass on the storage modifier
            if let Some(ref ty) = modifiers.storage {
                attributes.push(quote! { storage(#ty) });
            };
            // Pass on the cycle_delay_bug modifier
            if modifiers.cycle_delay_bug {
                attributes.push(quote! { cycle_delay_bug });
            };
            // Pass on the no_hash modifier
            if modifiers.no_hash {
                attributes.push(quote! { no_hash });
            };
            // Pass on the anon modifier
            if modifiers.anon {
                attributes.push(quote! { anon });
            };
            // Pass on the eval_always modifier
            if modifiers.eval_always {
                attributes.push(quote! { eval_always });
            };

            let attribute_stream = quote! {#(#attributes),*};
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

            add_query_description_impl(&query, modifiers, &mut query_description_stream);
        }
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
        macro_rules! rustc_cached_queries {
            ($($macro:tt)*) => {
                $($macro)*(#cached_queries);
            }
        }

        #query_description_stream
    })
}
