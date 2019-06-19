use hir::{Resolution, Either};
use ra_syntax::AstNode;
use test_utils::tested_by;

use crate::completion::{Completions, CompletionContext};

pub(super) fn complete_path(acc: &mut Completions, ctx: &CompletionContext) {
    let path = match &ctx.path_prefix {
        Some(path) => path.clone(),
        _ => return,
    };
    let def = match ctx.analyzer.resolve_hir_path(ctx.db, &path).take_types() {
        Some(Resolution::Def(def)) => def,
        _ => return,
    };
    match def {
        hir::ModuleDef::Module(module) => {
            let module_scope = module.scope(ctx.db);
            for (name, res) in module_scope.entries() {
                if let Some(hir::ModuleDef::BuiltinType(..)) = res.def.as_ref().take_types() {
                    if ctx.use_item_syntax.is_some() {
                        tested_by!(dont_complete_primitive_in_use);
                        continue;
                    }
                }
                if Some(module) == ctx.module {
                    if let Some(import) = res.import {
                        if let Either::A(use_tree) = module.import_source(ctx.db, import) {
                            if use_tree.syntax().range().contains_inclusive(ctx.offset) {
                                // for `use self::foo<|>`, don't suggest `foo` as a completion
                                tested_by!(dont_complete_current_use);
                                continue;
                            }
                        }
                    }
                }
                acc.add_resolution(ctx, name.to_string(), &res.def.map(hir::Resolution::Def));
            }
        }
        hir::ModuleDef::Enum(e) => {
            for variant in e.variants(ctx.db) {
                acc.add_enum_variant(ctx, variant);
            }
        }
        hir::ModuleDef::Struct(s) => {
            let ty = s.ty(ctx.db);
            let krate = ctx.module.and_then(|m| m.krate(ctx.db));
            if let Some(krate) = krate {
                ty.iterate_impl_items(ctx.db, krate, |item| {
                    match item {
                        hir::ImplItem::Method(func) => {
                            let data = func.data(ctx.db);
                            if !data.has_self_param() {
                                acc.add_function(ctx, func);
                            }
                        }
                        hir::ImplItem::Const(ct) => acc.add_const(ctx, ct),
                        hir::ImplItem::TypeAlias(ty) => acc.add_type_alias(ctx, ty),
                    }
                    None::<()>
                });
            }
        }
        _ => return,
    };
}

#[cfg(test)]
mod tests {
    use test_utils::covers;

    use crate::completion::{CompletionKind, check_completion, do_completion};

    fn check_reference_completion(code: &str, expected_completions: &str) {
        check_completion(code, expected_completions, CompletionKind::Reference);
    }

    #[test]
    fn dont_complete_current_use() {
        covers!(dont_complete_current_use);
        let completions = do_completion(r"use self::foo<|>;", CompletionKind::Reference);
        assert!(completions.is_empty());
    }

    #[test]
    fn dont_complete_current_use_in_braces_with_glob() {
        let completions = do_completion(
            r"
            mod foo { pub struct S; }
            use self::{foo::*, bar<|>};
            ",
            CompletionKind::Reference,
        );
        assert_eq!(completions.len(), 2);
    }

    #[test]
    fn dont_complete_primitive_in_use() {
        covers!(dont_complete_primitive_in_use);
        let completions = do_completion(r"use self::<|>;", CompletionKind::BuiltinType);
        assert!(completions.is_empty());
    }

    #[test]
    fn completes_primitives() {
        let completions =
            do_completion(r"fn main() { let _: <|> = 92; }", CompletionKind::BuiltinType);
        assert_eq!(completions.len(), 17);
    }

    #[test]
    fn completes_mod_with_docs() {
        check_reference_completion(
            "mod_with_docs",
            r"
            use self::my<|>;

            /// Some simple
            /// docs describing `mod my`.
            mod my {
                struct Bar;
            }
            ",
        );
    }

    #[test]
    fn completes_use_item_starting_with_self() {
        check_reference_completion(
            "use_item_starting_with_self",
            r"
            use self::m::<|>;

            mod m {
                struct Bar;
            }
            ",
        );
    }

    #[test]
    fn completes_use_item_starting_with_crate() {
        check_reference_completion(
            "use_item_starting_with_crate",
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::Sp<|>
            ",
        );
    }

    #[test]
    fn completes_nested_use_tree() {
        check_reference_completion(
            "nested_use_tree",
            "
            //- /lib.rs
            mod foo;
            struct Spam;
            //- /foo.rs
            use crate::{Sp<|>};
            ",
        );
    }

    #[test]
    fn completes_deeply_nested_use_tree() {
        check_reference_completion(
            "deeply_nested_use_tree",
            "
            //- /lib.rs
            mod foo;
            pub mod bar {
                pub mod baz {
                    pub struct Spam;
                }
            }
            //- /foo.rs
            use crate::{bar::{baz::Sp<|>}};
            ",
        );
    }

    #[test]
    fn completes_enum_variant() {
        check_reference_completion(
            "enum_variant",
            "
            //- /lib.rs
            /// An enum
            enum E {
                /// Foo Variant
                Foo,
                /// Bar Variant with i32
                Bar(i32)
            }
            fn foo() { let _ = E::<|> }
            ",
        );
    }

    #[test]
    fn completes_enum_variant_with_details() {
        check_reference_completion(
            "enum_variant_with_details",
            "
            //- /lib.rs
            struct S { field: u32 }
            /// An enum
            enum E {
                /// Foo Variant (empty)
                Foo,
                /// Bar Variant with i32 and u32
                Bar(i32, u32),
                ///
                S(S),
            }
            fn foo() { let _ = E::<|> }
            ",
        );
    }

    #[test]
    fn completes_struct_associated_method() {
        check_reference_completion(
            "struct_associated_method",
            "
            //- /lib.rs
            /// A Struct
            struct S;

            impl S {
                /// An associated method
                fn m() { }
            }

            fn foo() { let _ = S::<|> }
            ",
        );
    }

    #[test]
    fn completes_struct_associated_const() {
        check_reference_completion(
            "struct_associated_const",
            "
            //- /lib.rs
            /// A Struct
            struct S;

            impl S {
                /// An associated const
                const C: i32 = 42;
            }

            fn foo() { let _ = S::<|> }
            ",
        );
    }

    #[test]
    fn completes_struct_associated_type() {
        check_reference_completion(
            "struct_associated_type",
            "
            //- /lib.rs
            /// A Struct
            struct S;

            impl S {
                /// An associated type
                type T = i32;
            }

            fn foo() { let _ = S::<|> }
            ",
        );
    }

    #[test]
    fn completes_use_paths_across_crates() {
        check_reference_completion(
            "completes_use_paths_across_crates",
            "
            //- /main.rs
            use foo::<|>;

            //- /foo/lib.rs
            pub mod bar {
                pub struct S;
            }
            ",
        );
    }
}
