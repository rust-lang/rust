//! This modules takes care of rendering various definitions as completion items.

use hir::{db::HirDatabase, Docs, HasSource, HirDisplay, ScopeDef, Ty, TypeWalk};
use join_to_string::join;
use ra_syntax::ast::NameOwner;
use test_utils::tested_by;

use crate::completion::{
    db, CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

use crate::display::{const_label, function_label, macro_label, type_label};

impl Completions {
    pub(crate) fn add_field(
        &mut self,
        ctx: &CompletionContext,
        field: hir::StructField,
        substs: &hir::Substs,
    ) {
        CompletionItem::new(
            CompletionKind::Reference,
            ctx.source_range(),
            field.name(ctx.db).to_string(),
        )
        .kind(CompletionItemKind::Field)
        .detail(field.ty(ctx.db).subst(substs).display(ctx.db).to_string())
        .set_documentation(field.docs(ctx.db))
        .add_to(self);
    }

    pub(crate) fn add_tuple_field(&mut self, ctx: &CompletionContext, field: usize, ty: &hir::Ty) {
        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), field.to_string())
            .kind(CompletionItemKind::Field)
            .detail(ty.display(ctx.db).to_string())
            .add_to(self);
    }

    pub(crate) fn add_resolution(
        &mut self,
        ctx: &CompletionContext,
        local_name: String,
        resolution: &ScopeDef,
    ) {
        use hir::ModuleDef::*;

        let completion_kind = match resolution {
            ScopeDef::ModuleDef(BuiltinType(..)) => CompletionKind::BuiltinType,
            _ => CompletionKind::Reference,
        };

        let kind = match resolution {
            ScopeDef::ModuleDef(Module(..)) => CompletionItemKind::Module,
            ScopeDef::ModuleDef(Function(func)) => {
                return self.add_function_with_name(ctx, Some(local_name), *func);
            }
            ScopeDef::ModuleDef(Adt(hir::Adt::Struct(_))) => CompletionItemKind::Struct,
            // FIXME: add CompletionItemKind::Union
            ScopeDef::ModuleDef(Adt(hir::Adt::Union(_))) => CompletionItemKind::Struct,
            ScopeDef::ModuleDef(Adt(hir::Adt::Enum(_))) => CompletionItemKind::Enum,

            ScopeDef::ModuleDef(EnumVariant(..)) => CompletionItemKind::EnumVariant,
            ScopeDef::ModuleDef(Const(..)) => CompletionItemKind::Const,
            ScopeDef::ModuleDef(Static(..)) => CompletionItemKind::Static,
            ScopeDef::ModuleDef(Trait(..)) => CompletionItemKind::Trait,
            ScopeDef::ModuleDef(TypeAlias(..)) => CompletionItemKind::TypeAlias,
            ScopeDef::ModuleDef(BuiltinType(..)) => CompletionItemKind::BuiltinType,
            ScopeDef::GenericParam(..) => CompletionItemKind::TypeParam,
            ScopeDef::LocalBinding(..) => CompletionItemKind::Binding,
            // (does this need its own kind?)
            ScopeDef::AdtSelfType(..) | ScopeDef::ImplSelfType(..) => CompletionItemKind::TypeParam,
            ScopeDef::MacroDef(mac) => {
                return self.add_macro(ctx, Some(local_name), *mac);
            }
            ScopeDef::Unknown => {
                return self.add(CompletionItem::new(
                    CompletionKind::Reference,
                    ctx.source_range(),
                    local_name,
                ));
            }
        };

        let docs = match resolution {
            ScopeDef::ModuleDef(Module(it)) => it.docs(ctx.db),
            ScopeDef::ModuleDef(Adt(it)) => it.docs(ctx.db),
            ScopeDef::ModuleDef(EnumVariant(it)) => it.docs(ctx.db),
            ScopeDef::ModuleDef(Const(it)) => it.docs(ctx.db),
            ScopeDef::ModuleDef(Static(it)) => it.docs(ctx.db),
            ScopeDef::ModuleDef(Trait(it)) => it.docs(ctx.db),
            ScopeDef::ModuleDef(TypeAlias(it)) => it.docs(ctx.db),
            _ => None,
        };

        let mut completion_item =
            CompletionItem::new(completion_kind, ctx.source_range(), local_name.clone());
        if let ScopeDef::LocalBinding(pat_id) = resolution {
            let ty = ctx
                .analyzer
                .type_of_pat_by_id(ctx.db, pat_id.clone())
                .filter(|t| t != &Ty::Unknown)
                .map(|t| t.display(ctx.db).to_string());
            completion_item = completion_item.set_detail(ty);
        };

        // If not an import, add parenthesis automatically.
        if ctx.is_path_type
            && !ctx.has_type_args
            && ctx.db.feature_flags.get("completion.insertion.add-call-parenthesis")
        {
            let generic_def: Option<hir::GenericDef> = match resolution {
                ScopeDef::ModuleDef(Adt(it)) => Some((*it).into()),
                ScopeDef::ModuleDef(TypeAlias(it)) => Some((*it).into()),
                _ => None,
            };
            if let Some(def) = generic_def {
                if has_non_default_type_params(def, ctx.db) {
                    tested_by!(inserts_angle_brackets_for_generics);
                    completion_item = completion_item
                        .lookup_by(local_name.clone())
                        .label(format!("{}<…>", local_name))
                        .insert_snippet(format!("{}<$0>", local_name));
                }
            }
        }

        completion_item.kind(kind).set_documentation(docs).add_to(self)
    }

    pub(crate) fn add_function(&mut self, ctx: &CompletionContext, func: hir::Function) {
        self.add_function_with_name(ctx, None, func)
    }

    pub(crate) fn add_macro(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        macro_: hir::MacroDef,
    ) {
        let ast_node = macro_.source(ctx.db).ast;
        if let Some(name) = name {
            let detail = macro_label(&ast_node);

            let macro_braces_to_insert = match name.as_str() {
                "vec" => "[$0]",
                _ => "($0)",
            };
            let macro_declaration = name + "!";

            let builder = CompletionItem::new(
                CompletionKind::Reference,
                ctx.source_range(),
                &macro_declaration,
            )
            .kind(CompletionItemKind::Macro)
            .set_documentation(macro_.docs(ctx.db))
            .detail(detail)
            .insert_snippet(macro_declaration + macro_braces_to_insert);

            self.add(builder);
        }
    }

    fn add_function_with_name(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        func: hir::Function,
    ) {
        let data = func.data(ctx.db);
        let name = name.unwrap_or_else(|| data.name().to_string());
        let ast_node = func.source(ctx.db).ast;
        let detail = function_label(&ast_node);

        let mut builder =
            CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.clone())
                .kind(if data.has_self_param() {
                    CompletionItemKind::Method
                } else {
                    CompletionItemKind::Function
                })
                .set_documentation(func.docs(ctx.db))
                .detail(detail);

        // Add `<>` for generic types
        if ctx.use_item_syntax.is_none()
            && !ctx.is_call
            && ctx.db.feature_flags.get("completion.insertion.add-call-parenthesis")
        {
            tested_by!(inserts_parens_for_function_calls);
            let (snippet, label) =
                if data.params().is_empty() || data.has_self_param() && data.params().len() == 1 {
                    (format!("{}()$0", data.name()), format!("{}()", name))
                } else {
                    (format!("{}($0)", data.name()), format!("{}(…)", name))
                };
            builder = builder.lookup_by(name.clone()).label(label).insert_snippet(snippet);
        }

        self.add(builder)
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext, constant: hir::Const) {
        let ast_node = constant.source(ctx.db).ast;
        let name = match ast_node.name() {
            Some(name) => name,
            _ => return,
        };
        let detail = const_label(&ast_node);

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.text().to_string())
            .kind(CompletionItemKind::Const)
            .set_documentation(constant.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }

    pub(crate) fn add_type_alias(&mut self, ctx: &CompletionContext, type_alias: hir::TypeAlias) {
        let type_def = type_alias.source(ctx.db).ast;
        let name = match type_def.name() {
            Some(name) => name,
            _ => return,
        };
        let detail = type_label(&type_def);

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.text().to_string())
            .kind(CompletionItemKind::TypeAlias)
            .set_documentation(type_alias.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }

    pub(crate) fn add_enum_variant(&mut self, ctx: &CompletionContext, variant: hir::EnumVariant) {
        let name = match variant.name(ctx.db) {
            Some(it) => it,
            None => return,
        };
        let detail_types = variant.fields(ctx.db).into_iter().map(|field| field.ty(ctx.db));
        let detail = join(detail_types.map(|t| t.display(ctx.db).to_string()))
            .separator(", ")
            .surround_with("(", ")")
            .to_string();
        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.to_string())
            .kind(CompletionItemKind::EnumVariant)
            .set_documentation(variant.docs(ctx.db))
            .detail(detail)
            .add_to(self);
    }
}

fn has_non_default_type_params(def: hir::GenericDef, db: &db::RootDatabase) -> bool {
    let subst = db.generic_defaults(def);
    subst.iter().any(|ty| ty == &Ty::Unknown)
}

#[cfg(test)]
mod tests {
    use crate::completion::{do_completion, CompletionItem, CompletionKind};
    use insta::assert_debug_snapshot;
    use test_utils::covers;

    fn do_reference_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn inserts_parens_for_function_calls() {
        covers!(inserts_parens_for_function_calls);
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn no_args() {}
                fn main() { no_<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "main()",
                source_range: [61; 64),
                delete: [61; 64),
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "no_args()",
                source_range: [61; 64),
                delete: [61; 64),
                insert: "no_args()$0",
                kind: Function,
                lookup: "no_args",
                detail: "fn no_args()",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn with_args(x: i32, y: String) {}
                fn main() { with_<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "main()",
                source_range: [80; 85),
                delete: [80; 85),
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "with_args(…)",
                source_range: [80; 85),
                delete: [80; 85),
                insert: "with_args($0)",
                kind: Function,
                lookup: "with_args",
                detail: "fn with_args(x: i32, y: String)",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct S {}
                impl S {
                    fn foo(&self) {}
                }
                fn bar(s: &S) {
                    s.f<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo()",
                source_range: [163; 164),
                delete: [163; 164),
                insert: "foo()$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(&self)",
            },
        ]
        "###
        );
    }

    #[test]
    fn dont_render_function_parens_in_use_item() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                mod m { pub fn foo() {} }
                use crate::m::f<|>;
                "
            ),
            @r#"[
    CompletionItem {
        label: "foo",
        source_range: [40; 41),
        delete: [40; 41),
        insert: "foo",
        kind: Function,
        detail: "pub fn foo()",
    },
]"#
        );
    }

    #[test]
    fn dont_render_function_parens_if_already_call() {
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                fn frobnicate() {}
                fn main() {
                    frob<|>();
                }
                "
            ),
            @r#"[
    CompletionItem {
        label: "frobnicate",
        source_range: [35; 39),
        delete: [35; 39),
        insert: "frobnicate",
        kind: Function,
        detail: "fn frobnicate()",
    },
    CompletionItem {
        label: "main",
        source_range: [35; 39),
        delete: [35; 39),
        insert: "main",
        kind: Function,
        detail: "fn main()",
    },
]"#
        );
        assert_debug_snapshot!(
            do_reference_completion(
                "
                //- /lib.rs
                struct Foo {}
                impl Foo { fn new() -> Foo {} }
                fn main() {
                    Foo::ne<|>();
                }
                "
            ),
            @r#"[
    CompletionItem {
        label: "new",
        source_range: [67; 69),
        delete: [67; 69),
        insert: "new",
        kind: Function,
        detail: "fn new() -> Foo",
    },
]"#
        );
    }

    #[test]
    fn inserts_angle_brackets_for_generics() {
        covers!(inserts_angle_brackets_for_generics);
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct Vec<T> {}
                fn foo(xs: Ve<|>)
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Vec<…>",
                source_range: [61; 63),
                delete: [61; 63),
                insert: "Vec<$0>",
                kind: Struct,
                lookup: "Vec",
            },
            CompletionItem {
                label: "foo(…)",
                source_range: [61; 63),
                delete: [61; 63),
                insert: "foo($0)",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve)",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                type Vec<T> = (T,);
                fn foo(xs: Ve<|>)
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Vec<…>",
                source_range: [64; 66),
                delete: [64; 66),
                insert: "Vec<$0>",
                kind: TypeAlias,
                lookup: "Vec",
            },
            CompletionItem {
                label: "foo(…)",
                source_range: [64; 66),
                delete: [64; 66),
                insert: "foo($0)",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve)",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct Vec<T = i128> {}
                fn foo(xs: Ve<|>)
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Vec",
                source_range: [68; 70),
                delete: [68; 70),
                insert: "Vec",
                kind: Struct,
            },
            CompletionItem {
                label: "foo(…)",
                source_range: [68; 70),
                delete: [68; 70),
                insert: "foo($0)",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve)",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct Vec<T> {}
                fn foo(xs: Ve<|><i128>)
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "Vec",
                source_range: [61; 63),
                delete: [61; 63),
                insert: "Vec",
                kind: Struct,
            },
            CompletionItem {
                label: "foo(…)",
                source_range: [61; 63),
                delete: [61; 63),
                insert: "foo($0)",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve<i128>)",
            },
        ]
        "###
        );
    }
}
