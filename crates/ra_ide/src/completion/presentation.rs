//! This modules takes care of rendering various definitions as completion items.

use hir::{db::HirDatabase, Docs, HasAttrs, HasSource, HirDisplay, ScopeDef, Type};
use join_to_string::join;
use ra_syntax::ast::NameOwner;
use test_utils::tested_by;

use crate::completion::{
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, Completions,
};

use crate::display::{const_label, function_label, macro_label, type_label};

impl Completions {
    pub(crate) fn add_field(
        &mut self,
        ctx: &CompletionContext,
        field: hir::StructField,
        ty: &Type,
    ) {
        let is_deprecated = is_deprecated(field, ctx.db);
        CompletionItem::new(
            CompletionKind::Reference,
            ctx.source_range(),
            field.name(ctx.db).to_string(),
        )
        .kind(CompletionItemKind::Field)
        .detail(ty.display(ctx.db).to_string())
        .set_documentation(field.docs(ctx.db))
        .set_deprecated(is_deprecated)
        .add_to(self);
    }

    pub(crate) fn add_tuple_field(&mut self, ctx: &CompletionContext, field: usize, ty: &Type) {
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
            ScopeDef::Local(..) => CompletionItemKind::Binding,
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
        if let ScopeDef::Local(local) = resolution {
            let ty = local.ty(ctx.db);
            if !ty.is_unknown() {
                completion_item = completion_item.detail(ty.display(ctx.db).to_string());
            }
        };

        // If not an import, add parenthesis automatically.
        if ctx.is_path_type
            && !ctx.has_type_args
            && ctx.db.feature_flags.get("completion.insertion.add-call-parenthesis")
        {
            let has_non_default_type_params = match resolution {
                ScopeDef::ModuleDef(Adt(it)) => it.has_non_default_type_params(ctx.db),
                ScopeDef::ModuleDef(TypeAlias(it)) => it.has_non_default_type_params(ctx.db),
                _ => false,
            };
            if has_non_default_type_params {
                tested_by!(inserts_angle_brackets_for_generics);
                completion_item = completion_item
                    .lookup_by(local_name.clone())
                    .label(format!("{}<…>", local_name))
                    .insert_snippet(format!("{}<$0>", local_name));
            }
        }

        completion_item.kind(kind).set_documentation(docs).add_to(self)
    }

    pub(crate) fn add_function(&mut self, ctx: &CompletionContext, func: hir::Function) {
        self.add_function_with_name(ctx, None, func)
    }

    fn guess_macro_braces(&self, macro_name: &str, docs: &str) -> &'static str {
        let mut votes = [0, 0, 0];
        for (idx, s) in docs.match_indices(&macro_name) {
            let (before, after) = (&docs[..idx], &docs[idx + s.len()..]);
            // Ensure to match the full word
            if after.starts_with('!')
                && before
                    .chars()
                    .rev()
                    .next()
                    .map_or(true, |c| c != '_' && !c.is_ascii_alphanumeric())
            {
                // It may have spaces before the braces like `foo! {}`
                match after[1..].chars().find(|&c| !c.is_whitespace()) {
                    Some('{') => votes[0] += 1,
                    Some('[') => votes[1] += 1,
                    Some('(') => votes[2] += 1,
                    _ => {}
                }
            }
        }

        // Insert a space before `{}`.
        // We prefer the last one when some votes equal.
        *votes.iter().zip(&[" {$0}", "[$0]", "($0)"]).max_by_key(|&(&vote, _)| vote).unwrap().1
    }

    pub(crate) fn add_macro(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        macro_: hir::MacroDef,
    ) {
        let name = match name {
            Some(it) => it,
            None => return,
        };

        let ast_node = macro_.source(ctx.db).value;
        let detail = macro_label(&ast_node);

        let docs = macro_.docs(ctx.db);
        let macro_declaration = format!("{}!", name);

        let mut builder =
            CompletionItem::new(CompletionKind::Reference, ctx.source_range(), &macro_declaration)
                .kind(CompletionItemKind::Macro)
                .set_documentation(docs.clone())
                .set_deprecated(is_deprecated(macro_, ctx.db))
                .detail(detail);

        builder = if ctx.use_item_syntax.is_some() {
            builder.insert_text(name)
        } else {
            let macro_braces_to_insert =
                self.guess_macro_braces(&name, docs.as_ref().map_or("", |s| s.as_str()));
            builder.insert_snippet(macro_declaration + macro_braces_to_insert)
        };

        self.add(builder);
    }

    fn add_function_with_name(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        func: hir::Function,
    ) {
        let func_name = func.name(ctx.db);
        let has_self_param = func.has_self_param(ctx.db);
        let params = func.params(ctx.db);

        let name = name.unwrap_or_else(|| func_name.to_string());
        let ast_node = func.source(ctx.db).value;
        let detail = function_label(&ast_node);

        let mut builder =
            CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.clone())
                .kind(if has_self_param {
                    CompletionItemKind::Method
                } else {
                    CompletionItemKind::Function
                })
                .set_documentation(func.docs(ctx.db))
                .set_deprecated(is_deprecated(func, ctx.db))
                .detail(detail);

        // Add `<>` for generic types
        if ctx.use_item_syntax.is_none()
            && !ctx.is_call
            && ctx.db.feature_flags.get("completion.insertion.add-call-parenthesis")
        {
            tested_by!(inserts_parens_for_function_calls);
            let (snippet, label) = if params.is_empty() || has_self_param && params.len() == 1 {
                (format!("{}()$0", func_name), format!("{}()", name))
            } else {
                (format!("{}($0)", func_name), format!("{}(…)", name))
            };
            builder = builder.lookup_by(name).label(label).insert_snippet(snippet);
        }

        self.add(builder)
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext, constant: hir::Const) {
        let ast_node = constant.source(ctx.db).value;
        let name = match ast_node.name() {
            Some(name) => name,
            _ => return,
        };
        let detail = const_label(&ast_node);

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.text().to_string())
            .kind(CompletionItemKind::Const)
            .set_documentation(constant.docs(ctx.db))
            .set_deprecated(is_deprecated(constant, ctx.db))
            .detail(detail)
            .add_to(self);
    }

    pub(crate) fn add_type_alias(&mut self, ctx: &CompletionContext, type_alias: hir::TypeAlias) {
        let type_def = type_alias.source(ctx.db).value;
        let name = match type_def.name() {
            Some(name) => name,
            _ => return,
        };
        let detail = type_label(&type_def);

        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.text().to_string())
            .kind(CompletionItemKind::TypeAlias)
            .set_documentation(type_alias.docs(ctx.db))
            .set_deprecated(is_deprecated(type_alias, ctx.db))
            .detail(detail)
            .add_to(self);
    }

    pub(crate) fn add_enum_variant(&mut self, ctx: &CompletionContext, variant: hir::EnumVariant) {
        let is_deprecated = is_deprecated(variant, ctx.db);
        let name = variant.name(ctx.db);
        let detail_types = variant.fields(ctx.db).into_iter().map(|field| field.ty(ctx.db));
        let detail = join(detail_types.map(|t| t.display(ctx.db).to_string()))
            .separator(", ")
            .surround_with("(", ")")
            .to_string();
        CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.to_string())
            .kind(CompletionItemKind::EnumVariant)
            .set_documentation(variant.docs(ctx.db))
            .set_deprecated(is_deprecated)
            .detail(detail)
            .add_to(self);
    }
}

fn is_deprecated(node: impl HasAttrs, db: &impl HirDatabase) -> bool {
    node.attrs(db).by_key("deprecated").exists()
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use test_utils::covers;

    use crate::completion::{do_completion, CompletionItem, CompletionKind};

    fn do_reference_completion(code: &str) -> Vec<CompletionItem> {
        do_completion(code, CompletionKind::Reference)
    }

    #[test]
    fn sets_deprecated_flag_in_completion_items() {
        assert_debug_snapshot!(
            do_reference_completion(
                r#"
                #[deprecated]
                fn something_deprecated() {}

                #[deprecated(since = "1.0.0")]
                fn something_else_deprecated() {}

                fn main() { som<|> }
                "#,
            ),
            @r###"
        [
            CompletionItem {
                label: "main()",
                source_range: [203; 206),
                delete: [203; 206),
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "something_deprecated()",
                source_range: [203; 206),
                delete: [203; 206),
                insert: "something_deprecated()$0",
                kind: Function,
                lookup: "something_deprecated",
                detail: "fn something_deprecated()",
                deprecated: true,
            },
            CompletionItem {
                label: "something_else_deprecated()",
                source_range: [203; 206),
                delete: [203; 206),
                insert: "something_else_deprecated()$0",
                kind: Function,
                lookup: "something_else_deprecated",
                detail: "fn something_else_deprecated()",
                deprecated: true,
            },
        ]
        "###
        );
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
            @r###"
        [
            CompletionItem {
                label: "foo",
                source_range: [40; 41),
                delete: [40; 41),
                insert: "foo",
                kind: Function,
                detail: "pub fn foo()",
            },
        ]
        "###
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
            @r###"
        [
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
        ]
        "###
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
            @r###"
        [
            CompletionItem {
                label: "new",
                source_range: [67; 69),
                delete: [67; 69),
                insert: "new",
                kind: Function,
                detail: "fn new() -> Foo",
            },
        ]
        "###
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

    #[test]
    fn dont_insert_macro_call_braces_in_use() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                //- /main.rs
                use foo::<|>;

                //- /foo/lib.rs
                #[macro_export]
                macro_rules frobnicate {
                    () => ()
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "frobnicate!",
                source_range: [9; 9),
                delete: [9; 9),
                insert: "frobnicate",
                kind: Macro,
                detail: "#[macro_export]\nmacro_rules! frobnicate",
            },
        ]
        "###
        )
    }
}
