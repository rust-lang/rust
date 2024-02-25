//!
use heck::ToSnakeCase;
use proc_macro::TokenStream;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{Ident, ItemStruct, Path, Token};

type PunctuatedQueryGroups = Punctuated<QueryGroup, Token![,]>;

pub(crate) fn database(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = syn::parse_macro_input!(args as QueryGroupList);
    let input = syn::parse_macro_input!(input as ItemStruct);

    let query_groups = &args.query_groups;
    let database_name = &input.ident;
    let visibility = &input.vis;
    let db_storage_field = quote! { storage };

    let mut output = proc_macro2::TokenStream::new();
    output.extend(quote! { #input });

    let query_group_names_snake: Vec<_> = query_groups
        .iter()
        .map(|query_group| {
            let group_name = query_group.name();
            Ident::new(&group_name.to_string().to_snake_case(), group_name.span())
        })
        .collect();

    let query_group_storage_names: Vec<_> = query_groups
        .iter()
        .map(|QueryGroup { group_path }| {
            quote! {
                <#group_path as salsa::plumbing::QueryGroup>::GroupStorage
            }
        })
        .collect();

    // For each query group `foo::MyGroup` create a link to its
    // `foo::MyGroupGroupStorage`
    let mut storage_fields = proc_macro2::TokenStream::new();
    let mut storage_initializers = proc_macro2::TokenStream::new();
    let mut has_group_impls = proc_macro2::TokenStream::new();
    for (((query_group, group_name_snake), group_storage), group_index) in query_groups
        .iter()
        .zip(&query_group_names_snake)
        .zip(&query_group_storage_names)
        .zip(0_u16..)
    {
        let group_path = &query_group.group_path;

        // rewrite the last identifier (`MyGroup`, above) to
        // (e.g.) `MyGroupGroupStorage`.
        storage_fields.extend(quote! {
            #group_name_snake: #group_storage,
        });

        // rewrite the last identifier (`MyGroup`, above) to
        // (e.g.) `MyGroupGroupStorage`.
        storage_initializers.extend(quote! {
            #group_name_snake: #group_storage::new(#group_index),
        });

        // ANCHOR:HasQueryGroup
        has_group_impls.extend(quote! {
            impl salsa::plumbing::HasQueryGroup<#group_path> for #database_name {
                fn group_storage(&self) -> &#group_storage {
                    &self.#db_storage_field.query_store().#group_name_snake
                }

                fn group_storage_mut(&mut self) -> (&#group_storage, &mut salsa::Runtime) {
                    let (query_store_mut, runtime) = self.#db_storage_field.query_store_mut();
                    (&query_store_mut.#group_name_snake, runtime)
                }
            }
        });
        // ANCHOR_END:HasQueryGroup
    }

    // create group storage wrapper struct
    output.extend(quote! {
        #[doc(hidden)]
        #visibility struct __SalsaDatabaseStorage {
            #storage_fields
        }

        impl Default for __SalsaDatabaseStorage {
            fn default() -> Self {
                Self {
                    #storage_initializers
                }
            }
        }
    });

    // Create a tuple (D1, D2, ...) where Di is the data for a given query group.
    let mut database_data = vec![];
    for QueryGroup { group_path } in query_groups {
        database_data.push(quote! {
            <#group_path as salsa::plumbing::QueryGroup>::GroupData
        });
    }

    // ANCHOR:DatabaseStorageTypes
    output.extend(quote! {
        impl salsa::plumbing::DatabaseStorageTypes for #database_name {
            type DatabaseStorage = __SalsaDatabaseStorage;
        }
    });
    // ANCHOR_END:DatabaseStorageTypes

    // ANCHOR:DatabaseOps
    let mut fmt_ops = proc_macro2::TokenStream::new();
    let mut maybe_changed_ops = proc_macro2::TokenStream::new();
    let mut cycle_recovery_strategy_ops = proc_macro2::TokenStream::new();
    let mut for_each_ops = proc_macro2::TokenStream::new();
    for ((QueryGroup { group_path }, group_storage), group_index) in
        query_groups.iter().zip(&query_group_storage_names).zip(0_u16..)
    {
        fmt_ops.extend(quote! {
            #group_index => {
                let storage: &#group_storage =
                    <Self as salsa::plumbing::HasQueryGroup<#group_path>>::group_storage(self);
                storage.fmt_index(self, input, fmt)
            }
        });
        maybe_changed_ops.extend(quote! {
            #group_index => {
                let storage: &#group_storage =
                    <Self as salsa::plumbing::HasQueryGroup<#group_path>>::group_storage(self);
                storage.maybe_changed_after(self, input, revision)
            }
        });
        cycle_recovery_strategy_ops.extend(quote! {
            #group_index => {
                let storage: &#group_storage =
                    <Self as salsa::plumbing::HasQueryGroup<#group_path>>::group_storage(self);
                storage.cycle_recovery_strategy(self, input)
            }
        });
        for_each_ops.extend(quote! {
            let storage: &#group_storage =
                <Self as salsa::plumbing::HasQueryGroup<#group_path>>::group_storage(self);
            storage.for_each_query(runtime, &mut op);
        });
    }
    output.extend(quote! {
        impl salsa::plumbing::DatabaseOps for #database_name {
            fn ops_database(&self) -> &dyn salsa::Database {
                self
            }

            fn ops_salsa_runtime(&self) -> &salsa::Runtime {
                self.#db_storage_field.salsa_runtime()
            }

            fn synthetic_write(&mut self, durability: salsa::Durability) {
                self.#db_storage_field.salsa_runtime_mut().synthetic_write(durability)
            }

            fn fmt_index(
                &self,
                input: salsa::DatabaseKeyIndex,
                fmt: &mut std::fmt::Formatter<'_>,
            ) -> std::fmt::Result {
                match input.group_index() {
                    #fmt_ops
                    i => panic!("salsa: invalid group index {}", i)
                }
            }

            fn maybe_changed_after(
                &self,
                input: salsa::DatabaseKeyIndex,
                revision: salsa::Revision
            ) -> bool {
                match input.group_index() {
                    #maybe_changed_ops
                    i => panic!("salsa: invalid group index {}", i)
                }
            }

            fn cycle_recovery_strategy(
                &self,
                input: salsa::DatabaseKeyIndex,
            ) -> salsa::plumbing::CycleRecoveryStrategy {
                match input.group_index() {
                    #cycle_recovery_strategy_ops
                    i => panic!("salsa: invalid group index {}", i)
                }
            }

            fn for_each_query(
                &self,
                mut op: &mut dyn FnMut(&dyn salsa::plumbing::QueryStorageMassOps),
            ) {
                let runtime = salsa::Database::salsa_runtime(self);
                #for_each_ops
            }
        }
    });
    // ANCHOR_END:DatabaseOps

    output.extend(has_group_impls);

    output.into()
}

#[derive(Clone, Debug)]
struct QueryGroupList {
    query_groups: PunctuatedQueryGroups,
}

impl Parse for QueryGroupList {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let query_groups: PunctuatedQueryGroups =
            input.parse_terminated(QueryGroup::parse, Token![,])?;
        Ok(QueryGroupList { query_groups })
    }
}

#[derive(Clone, Debug)]
struct QueryGroup {
    group_path: Path,
}

impl QueryGroup {
    /// The name of the query group trait.
    fn name(&self) -> Ident {
        self.group_path.segments.last().unwrap().ident.clone()
    }
}

impl Parse for QueryGroup {
    /// ```ignore
    ///         impl HelloWorldDatabase;
    /// ```
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let group_path: Path = input.parse()?;
        Ok(QueryGroup { group_path })
    }
}

struct Nothing;

impl Parse for Nothing {
    fn parse(_input: ParseStream<'_>) -> syn::Result<Self> {
        Ok(Nothing)
    }
}
