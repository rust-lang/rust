use proc_macro::TokenStream;
use proc_macro2::{TokenTree, Delimiter};
use syn::{
    Token, Ident, Type, Attribute, ReturnType, Expr, Block, Error,
    braced, parenthesized, parse_macro_input,
};
use syn::spanned::Spanned;
use syn::parse::{Result, Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn;
use quote::quote;
use itertools::Itertools;

#[allow(non_camel_case_types)]
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

    /// Don't force the query
    NoForce,

    /// Generate a dep node based on the dependencies of the query
    Anon,

    /// Always evaluate the query, ignoring its depdendencies
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
        } else if modifier == "fatal_cycle" {
            Ok(QueryModifier::FatalCycle)
        } else if modifier == "cycle_delay_bug" {
            Ok(QueryModifier::CycleDelayBug)
        } else if modifier == "no_hash" {
            Ok(QueryModifier::NoHash)
        } else if modifier == "no_force" {
            Ok(QueryModifier::NoForce)
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
fn check_attributes(attrs: Vec<Attribute>) -> Result<()> {
    for attr in attrs {
        if !attr.path.is_ident("doc") {
            return Err(Error::new(attr.span(), "attributes not supported on queries"));
        }
    }
    Ok(())
}

/// A compiler query. `query ... { ... }`
struct Query {
    modifiers: List<QueryModifier>,
    name: Ident,
    key: IdentOrWild,
    arg: Type,
    result: ReturnType,
}

impl Parse for Query {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        check_attributes(input.call(Attribute::parse_outer)?)?;

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

        Ok(Query {
            modifiers,
            name,
            key,
            arg,
            result,
        })
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

struct QueryModifiers {
    /// The description of the query.
    desc: Option<(Option<Ident>, Punctuated<Expr, Token![,]>)>,

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

    /// Don't force the query
    no_force: bool,

    /// Generate a dep node based on the dependencies of the query
    anon: bool,

    // Always evaluate the query, ignoring its depdendencies
    eval_always: bool,
}

/// Process query modifiers into a struct, erroring on duplicates
fn process_modifiers(query: &mut Query) -> QueryModifiers {
    let mut load_cached = None;
    let mut cache = None;
    let mut desc = None;
    let mut fatal_cycle = false;
    let mut cycle_delay_bug = false;
    let mut no_hash = false;
    let mut no_force = false;
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
            QueryModifier::NoForce => {
                if no_force {
                    panic!("duplicate modifier `no_force` for query `{}`", query.name);
                }
                no_force = true;
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
    QueryModifiers {
        load_cached,
        cache,
        desc,
        fatal_cycle,
        cycle_delay_bug,
        no_hash,
        no_force,
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
    let cache = modifiers.cache.as_ref().map(|(args, expr)| {
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
                    tcx.queries.on_disk_cache.try_load_query_result(tcx, id)
                }
            }
        };

        let tcx = args.as_ref().map(|t| {
            let t = &(t.0).0;
            quote! { #t }
        }).unwrap_or(quote! { _ });
        let value = args.as_ref().map(|t| {
            let t = &(t.1).0;
            quote! { #t }
        }).unwrap_or(quote! { _ });
        quote! {
            #[inline]
            #[allow(unused_variables)]
            fn cache_on_disk(
                #tcx: TyCtxt<'tcx>,
                #key: Self::Key,
                #value: Option<&Self::Value>
            ) -> bool {
                #expr
            }

            #try_load_from_disk
        }
    });

    if cache.is_none() && modifiers.load_cached.is_some() {
        panic!("load_cached modifier on query `{}` without a cache modifier", name);
    }

    let desc = modifiers.desc.as_ref().map(|(tcx, desc)| {
        let tcx = tcx.as_ref().map(|t| quote! { #t }).unwrap_or(quote! { _ });
        quote! {
            #[allow(unused_variables)]
            fn describe(
                #tcx: TyCtxt<'_>,
                #key: #arg,
            ) -> Cow<'static, str> {
                format!(#desc).into()
            }
        }
    });

    if desc.is_some() || cache.is_some() {
        let cache = cache.unwrap_or(quote! {});
        let desc = desc.unwrap_or(quote! {});

        impls.extend(quote! {
            impl<'tcx> QueryDescription<'tcx> for queries::#name<'tcx> {
                #desc
                #cache
            }
        });
    }
}

pub fn rustc_queries(input: TokenStream) -> TokenStream {
    let groups = parse_macro_input!(input as List<Group>);

    let mut query_stream = quote! {};
    let mut query_description_stream = quote! {};
    let mut dep_node_def_stream = quote! {};
    let mut dep_node_force_stream = quote! {};
    let mut try_load_from_on_disk_cache_stream = quote! {};
    let mut no_force_queries = Vec::new();
    let mut cached_queries = quote! {};

    for group in groups.0 {
        let mut group_stream = quote! {};
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

            if modifiers.cache.is_some() && !modifiers.no_force {
                try_load_from_on_disk_cache_stream.extend(quote! {
                    DepKind::#name => {
                        debug_assert!(tcx.dep_graph
                                         .node_color(self)
                                         .map(|c| c.is_green())
                                         .unwrap_or(false));

                        let key = RecoverKey::recover(tcx.global_tcx(), self).unwrap();
                        if queries::#name::cache_on_disk(tcx.global_tcx(), key, None) {
                            let _ = tcx.#name(key);
                        }
                    }
                });
            }

            let mut attributes = Vec::new();

            // Pass on the fatal_cycle modifier
            if modifiers.fatal_cycle {
                attributes.push(quote! { fatal_cycle });
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

            let mut attribute_stream = quote! {};
            for e in attributes.into_iter().intersperse(quote! {,}) {
                attribute_stream.extend(e);
            }

            // Add the query to the group
            group_stream.extend(quote! {
                [#attribute_stream] fn #name: #name(#arg) #result,
            });

            // Create a dep node for the query
            dep_node_def_stream.extend(quote! {
                [#attribute_stream] #name(#arg),
            });

            if modifiers.no_force {
                no_force_queries.push(name.clone());
            } else {
                // Add a match arm to force the query given the dep node
                dep_node_force_stream.extend(quote! {
                    DepKind::#name => {
                        if let Some(key) = RecoverKey::recover($tcx, $dep_node) {
                            force_ex!($tcx, #name, key);
                        } else {
                            return false;
                        }
                    }
                });
            }

            add_query_description_impl(
                &query,
                modifiers,
                &mut query_description_stream,
            );
        }
        let name = &group.name;
        query_stream.extend(quote! {
            #name { #group_stream },
        });
    }

    // Add an arm for the no force queries to panic when trying to force them
    for query in no_force_queries {
        dep_node_force_stream.extend(quote! {
            DepKind::#query |
        });
    }
    dep_node_force_stream.extend(quote! {
        DepKind::Null => {
            bug!("Cannot force dep node: {:?}", $dep_node)
        }
    });

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
        macro_rules! rustc_cached_queries {
            ($($macro:tt)*) => {
                $($macro)*(#cached_queries);
            }
        }

        #query_description_stream

        impl DepNode {
            /// Check whether the query invocation corresponding to the given
            /// DepNode is eligible for on-disk-caching. If so, this is method
            /// will execute the query corresponding to the given DepNode.
            /// Also, as a sanity check, it expects that the corresponding query
            /// invocation has been marked as green already.
            pub fn try_load_from_on_disk_cache(&self, tcx: TyCtxt<'_>) {
                match self.kind {
                    #try_load_from_on_disk_cache_stream
                    _ => (),
                }
            }
        }
    })
}
