//! This modules takes care of rendering various definitions as completion items.

use hir::{Docs, HasAttrs, HasSource, HirDisplay, ModPath, ScopeDef, StructKind, Type};
use ra_syntax::ast::NameOwner;
use stdx::SepBy;
use test_utils::mark;

use crate::{
    completion::{
        completion_item::Builder, CompletionContext, CompletionItem, CompletionItemKind,
        CompletionKind, Completions,
    },
    display::{const_label, macro_label, type_label, FunctionSignature},
    CompletionScore, RootDatabase,
};

impl Completions {
    pub(crate) fn add_field(&mut self, ctx: &CompletionContext, field: hir::Field, ty: &Type) {
        let is_deprecated = is_deprecated(field, ctx.db);
        let name = field.name(ctx.db);
        let mut completion_item =
            CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.to_string())
                .kind(CompletionItemKind::Field)
                .detail(ty.display(ctx.db).to_string())
                .set_documentation(field.docs(ctx.db))
                .set_deprecated(is_deprecated);

        if let Some(score) = compute_score(ctx, &ty, &name.to_string()) {
            completion_item = completion_item.set_score(score);
        }

        completion_item.add_to(self);
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
                return self.add_function(ctx, *func, Some(local_name));
            }
            ScopeDef::ModuleDef(Adt(hir::Adt::Struct(_))) => CompletionItemKind::Struct,
            // FIXME: add CompletionItemKind::Union
            ScopeDef::ModuleDef(Adt(hir::Adt::Union(_))) => CompletionItemKind::Struct,
            ScopeDef::ModuleDef(Adt(hir::Adt::Enum(_))) => CompletionItemKind::Enum,

            ScopeDef::ModuleDef(EnumVariant(var)) => {
                return self.add_enum_variant(ctx, *var, Some(local_name));
            }
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

        if let ScopeDef::Local(local) = resolution {
            if let Some(score) = compute_score(ctx, &local.ty(ctx.db), &local_name) {
                completion_item = completion_item.set_score(score);
            }
        }

        // Add `<>` for generic types
        if ctx.is_path_type && !ctx.has_type_args && ctx.config.add_call_parenthesis {
            if let Some(cap) = ctx.config.snippet_cap {
                let has_non_default_type_params = match resolution {
                    ScopeDef::ModuleDef(Adt(it)) => it.has_non_default_type_params(ctx.db),
                    ScopeDef::ModuleDef(TypeAlias(it)) => it.has_non_default_type_params(ctx.db),
                    _ => false,
                };
                if has_non_default_type_params {
                    mark::hit!(inserts_angle_brackets_for_generics);
                    completion_item = completion_item
                        .lookup_by(local_name.clone())
                        .label(format!("{}<…>", local_name))
                        .insert_snippet(cap, format!("{}<$0>", local_name));
                }
            }
        }

        completion_item.kind(kind).set_documentation(docs).add_to(self)
    }

    pub(crate) fn add_macro(
        &mut self,
        ctx: &CompletionContext,
        name: Option<String>,
        macro_: hir::MacroDef,
    ) {
        // FIXME: Currently proc-macro do not have ast-node,
        // such that it does not have source
        if macro_.is_proc_macro() {
            return;
        }

        let name = match name {
            Some(it) => it,
            None => return,
        };

        let ast_node = macro_.source(ctx.db).value;
        let detail = macro_label(&ast_node);

        let docs = macro_.docs(ctx.db);

        let mut builder = CompletionItem::new(
            CompletionKind::Reference,
            ctx.source_range(),
            &format!("{}!", name),
        )
        .kind(CompletionItemKind::Macro)
        .set_documentation(docs.clone())
        .set_deprecated(is_deprecated(macro_, ctx.db))
        .detail(detail);

        let needs_bang = ctx.use_item_syntax.is_none() && !ctx.is_macro_call;
        builder = match ctx.config.snippet_cap {
            Some(cap) if needs_bang => {
                let docs = docs.as_ref().map_or("", |s| s.as_str());
                let (bra, ket) = guess_macro_braces(&name, docs);
                builder
                    .insert_snippet(cap, format!("{}!{}$0{}", name, bra, ket))
                    .label(format!("{}!{}…{}", name, bra, ket))
            }
            None if needs_bang => builder.insert_text(format!("{}!", name)),
            _ => {
                mark::hit!(dont_insert_macro_call_parens_unncessary);
                builder.insert_text(name)
            }
        };

        self.add(builder);
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: &CompletionContext,
        func: hir::Function,
        local_name: Option<String>,
    ) {
        let has_self_param = func.has_self_param(ctx.db);

        let name = local_name.unwrap_or_else(|| func.name(ctx.db).to_string());
        let ast_node = func.source(ctx.db).value;
        let function_signature = FunctionSignature::from(&ast_node);

        let mut builder =
            CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.clone())
                .kind(if has_self_param {
                    CompletionItemKind::Method
                } else {
                    CompletionItemKind::Function
                })
                .set_documentation(func.docs(ctx.db))
                .set_deprecated(is_deprecated(func, ctx.db))
                .detail(function_signature.to_string());

        let params = function_signature
            .parameter_names
            .iter()
            .skip(if function_signature.has_self_param { 1 } else { 0 })
            .map(|name| name.trim_start_matches('_').into())
            .collect();

        builder = builder.add_call_parens(ctx, name, Params::Named(params));

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

    pub(crate) fn add_qualified_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::EnumVariant,
        path: ModPath,
    ) {
        self.add_enum_variant_impl(ctx, variant, None, Some(path))
    }

    pub(crate) fn add_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::EnumVariant,
        local_name: Option<String>,
    ) {
        self.add_enum_variant_impl(ctx, variant, local_name, None)
    }

    fn add_enum_variant_impl(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::EnumVariant,
        local_name: Option<String>,
        path: Option<ModPath>,
    ) {
        let is_deprecated = is_deprecated(variant, ctx.db);
        let name = local_name.unwrap_or_else(|| variant.name(ctx.db).to_string());
        let qualified_name = match &path {
            Some(it) => it.to_string(),
            None => name.to_string(),
        };
        let detail_types = variant
            .fields(ctx.db)
            .into_iter()
            .map(|field| (field.name(ctx.db), field.signature_ty(ctx.db)));
        let variant_kind = variant.kind(ctx.db);
        let detail = match variant_kind {
            StructKind::Tuple | StructKind::Unit => detail_types
                .map(|(_, t)| t.display(ctx.db).to_string())
                .sep_by(", ")
                .surround_with("(", ")")
                .to_string(),
            StructKind::Record => detail_types
                .map(|(n, t)| format!("{}: {}", n, t.display(ctx.db).to_string()))
                .sep_by(", ")
                .surround_with("{ ", " }")
                .to_string(),
        };
        let mut res = CompletionItem::new(
            CompletionKind::Reference,
            ctx.source_range(),
            qualified_name.clone(),
        )
        .kind(CompletionItemKind::EnumVariant)
        .set_documentation(variant.docs(ctx.db))
        .set_deprecated(is_deprecated)
        .detail(detail);

        if path.is_some() {
            res = res.lookup_by(name);
        }

        if variant_kind == StructKind::Tuple {
            let params = Params::Anonymous(variant.fields(ctx.db).len());
            res = res.add_call_parens(ctx, qualified_name, params)
        }

        res.add_to(self);
    }
}

pub(crate) fn compute_score(
    ctx: &CompletionContext,
    ty: &Type,
    name: &str,
) -> Option<CompletionScore> {
    // FIXME: this should not fall back to string equality.
    let ty = &ty.display(ctx.db).to_string();
    let (active_name, active_type) = if let Some(record_field) = &ctx.record_field_syntax {
        mark::hit!(test_struct_field_completion_in_record_lit);
        let (struct_field, _local) = ctx.sema.resolve_record_field(record_field)?;
        (
            struct_field.name(ctx.db).to_string(),
            struct_field.signature_ty(ctx.db).display(ctx.db).to_string(),
        )
    } else if let Some(active_parameter) = &ctx.active_parameter {
        mark::hit!(test_struct_field_completion_in_func_call);
        (active_parameter.name.clone(), active_parameter.ty.clone())
    } else {
        return None;
    };

    // Compute score
    // For the same type
    if &active_type != ty {
        return None;
    }

    let mut res = CompletionScore::TypeMatch;

    // If same type + same name then go top position
    if active_name == name {
        res = CompletionScore::TypeAndNameMatch
    }

    Some(res)
}

enum Params {
    Named(Vec<String>),
    Anonymous(usize),
}

impl Params {
    fn len(&self) -> usize {
        match self {
            Params::Named(xs) => xs.len(),
            Params::Anonymous(len) => *len,
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Builder {
    fn add_call_parens(mut self, ctx: &CompletionContext, name: String, params: Params) -> Builder {
        if !ctx.config.add_call_parenthesis {
            return self;
        }
        if ctx.use_item_syntax.is_some() || ctx.is_call {
            return self;
        }

        // Don't add parentheses if the expected type is some function reference.
        if let Some(ty) = &ctx.expected_type {
            if ty.is_fn() {
                return self;
            }
        }

        let cap = match ctx.config.snippet_cap {
            Some(it) => it,
            None => return self,
        };
        // If not an import, add parenthesis automatically.
        mark::hit!(inserts_parens_for_function_calls);

        let (snippet, label) = if params.is_empty() {
            (format!("{}()$0", name), format!("{}()", name))
        } else {
            self = self.trigger_call_info();
            let snippet = match (ctx.config.add_call_argument_snippets, params) {
                (true, Params::Named(params)) => {
                    let function_params_snippet = params
                        .iter()
                        .enumerate()
                        .map(|(index, param_name)| format!("${{{}:{}}}", index + 1, param_name))
                        .sep_by(", ");
                    format!("{}({})$0", name, function_params_snippet)
                }
                _ => format!("{}($0)", name),
            };

            (snippet, format!("{}(…)", name))
        };
        self.lookup_by(name).label(label).insert_snippet(cap, snippet)
    }
}

fn is_deprecated(node: impl HasAttrs, db: &RootDatabase) -> bool {
    node.attrs(db).by_key("deprecated").exists()
}

fn guess_macro_braces(macro_name: &str, docs: &str) -> (&'static str, &'static str) {
    let mut votes = [0, 0, 0];
    for (idx, s) in docs.match_indices(&macro_name) {
        let (before, after) = (&docs[..idx], &docs[idx + s.len()..]);
        // Ensure to match the full word
        if after.starts_with('!')
            && !before.ends_with(|c: char| c == '_' || c.is_ascii_alphanumeric())
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
    let (_vote, (bra, ket)) = votes
        .iter()
        .zip(&[(" {", "}"), ("[", "]"), ("(", ")")])
        .max_by_key(|&(&vote, _)| vote)
        .unwrap();
    (*bra, *ket)
}

#[cfg(test)]
mod tests {
    use insta::assert_debug_snapshot;
    use test_utils::mark;

    use crate::completion::{
        test_utils::{do_completion, do_completion_with_options},
        CompletionConfig, CompletionItem, CompletionKind,
    };

    fn do_reference_completion(ra_fixture: &str) -> Vec<CompletionItem> {
        do_completion(ra_fixture, CompletionKind::Reference)
    }

    fn do_reference_completion_with_options(
        ra_fixture: &str,
        options: CompletionConfig,
    ) -> Vec<CompletionItem> {
        do_completion_with_options(ra_fixture, CompletionKind::Reference, &options)
    }

    #[test]
    fn enum_detail_includes_names_for_record() {
        assert_debug_snapshot!(
        do_reference_completion(
            r#"
                enum Foo {
                    Foo {x: i32, y: i32}
                }

                fn main() { Foo::Fo<|> }
                "#,
        ),
        @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 56..58,
                delete: 56..58,
                insert: "Foo",
                kind: EnumVariant,
                detail: "{ x: i32, y: i32 }",
            },
        ]
        "###
        );
    }

    #[test]
    fn enum_detail_doesnt_include_names_for_tuple() {
        assert_debug_snapshot!(
        do_reference_completion(
            r#"
                enum Foo {
                    Foo (i32, i32)
                }

                fn main() { Foo::Fo<|> }
                "#,
        ),
        @r###"
        [
            CompletionItem {
                label: "Foo(…)",
                source_range: 50..52,
                delete: 50..52,
                insert: "Foo($0)",
                kind: EnumVariant,
                lookup: "Foo",
                detail: "(i32, i32)",
                trigger_call_info: true,
            },
        ]
        "###
        );
    }

    #[test]
    fn enum_detail_just_parentheses_for_unit() {
        assert_debug_snapshot!(
        do_reference_completion(
            r#"
                enum Foo {
                    Foo
                }

                fn main() { Foo::Fo<|> }
                "#,
        ),
        @r###"
        [
            CompletionItem {
                label: "Foo",
                source_range: 39..41,
                delete: 39..41,
                insert: "Foo",
                kind: EnumVariant,
                detail: "()",
            },
        ]
        "###
        );
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
                source_range: 122..125,
                delete: 122..125,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "something_deprecated()",
                source_range: 122..125,
                delete: 122..125,
                insert: "something_deprecated()$0",
                kind: Function,
                lookup: "something_deprecated",
                detail: "fn something_deprecated()",
                deprecated: true,
            },
            CompletionItem {
                label: "something_else_deprecated()",
                source_range: 122..125,
                delete: 122..125,
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
        mark::check!(inserts_parens_for_function_calls);
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
                source_range: 28..31,
                delete: 28..31,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "no_args()",
                source_range: 28..31,
                delete: 28..31,
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
                source_range: 47..52,
                delete: 47..52,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "with_args(…)",
                source_range: 47..52,
                delete: 47..52,
                insert: "with_args(${1:x}, ${2:y})$0",
                kind: Function,
                lookup: "with_args",
                detail: "fn with_args(x: i32, y: String)",
                trigger_call_info: true,
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn with_ignored_args(_foo: i32, ___bar: bool, ho_ge_: String) {}
                fn main() { with_<|> }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "main()",
                source_range: 77..82,
                delete: 77..82,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
            CompletionItem {
                label: "with_ignored_args(…)",
                source_range: 77..82,
                delete: 77..82,
                insert: "with_ignored_args(${1:foo}, ${2:bar}, ${3:ho_ge_})$0",
                kind: Function,
                lookup: "with_ignored_args",
                detail: "fn with_ignored_args(_foo: i32, ___bar: bool, ho_ge_: String)",
                trigger_call_info: true,
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
                source_range: 66..67,
                delete: 66..67,
                insert: "foo()$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(&self)",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct S {}
                impl S {
                    fn foo_ignored_args(&self, _a: bool, b: i32) {}
                }
                fn bar(s: &S) {
                    s.f<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo_ignored_args(…)",
                source_range: 97..98,
                delete: 97..98,
                insert: "foo_ignored_args(${1:a}, ${2:b})$0",
                kind: Method,
                lookup: "foo_ignored_args",
                detail: "fn foo_ignored_args(&self, _a: bool, b: i32)",
                trigger_call_info: true,
            },
        ]
        "###
        );
    }

    #[test]
    fn inserts_parens_for_tuple_enums() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Option<T> { Some(T), None }
                use Option::*;
                fn main() -> Option<i32> {
                    Som<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "None",
                source_range: 79..82,
                delete: 79..82,
                insert: "None",
                kind: EnumVariant,
                detail: "()",
            },
            CompletionItem {
                label: "Option",
                source_range: 79..82,
                delete: 79..82,
                insert: "Option",
                kind: Enum,
            },
            CompletionItem {
                label: "Some(…)",
                source_range: 79..82,
                delete: 79..82,
                insert: "Some($0)",
                kind: EnumVariant,
                lookup: "Some",
                detail: "(T)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "main()",
                source_range: 79..82,
                delete: 79..82,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main() -> Option<i32>",
            },
        ]
        "###
        );
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                enum Option<T> { Some(T), None }
                use Option::*;
                fn main(value: Option<i32>) {
                    match value {
                        Som<|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "None",
                source_range: 104..107,
                delete: 104..107,
                insert: "None",
                kind: EnumVariant,
                detail: "()",
            },
            CompletionItem {
                label: "Option",
                source_range: 104..107,
                delete: 104..107,
                insert: "Option",
                kind: Enum,
            },
            CompletionItem {
                label: "Some(…)",
                source_range: 104..107,
                delete: 104..107,
                insert: "Some($0)",
                kind: EnumVariant,
                lookup: "Some",
                detail: "(T)",
                trigger_call_info: true,
            },
        ]
        "###
        );
    }

    #[test]
    fn no_call_parens_if_fn_ptr_needed() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                fn somefn(with: u8, a: u8, lot: u8, of: u8, args: u8) {}

                struct ManualVtable {
                    method: fn(u8, u8, u8, u8, u8),
                }

                fn main() -> ManualVtable {
                    ManualVtable {
                        method: some<|>
                    }
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "ManualVtable",
                source_range: 182..186,
                delete: 182..186,
                insert: "ManualVtable",
                kind: Struct,
            },
            CompletionItem {
                label: "main",
                source_range: 182..186,
                delete: 182..186,
                insert: "main",
                kind: Function,
                detail: "fn main() -> ManualVtable",
            },
            CompletionItem {
                label: "somefn",
                source_range: 182..186,
                delete: 182..186,
                insert: "somefn",
                kind: Function,
                detail: "fn somefn(with: u8, a: u8, lot: u8, of: u8, args: u8)",
            },
        ]
        "###
        );
    }

    #[test]
    fn arg_snippets_for_method_call() {
        assert_debug_snapshot!(
            do_reference_completion(
                r"
                struct S {}
                impl S {
                    fn foo(&self, x: i32) {}
                }
                fn bar(s: &S) {
                    s.f<|>
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "foo(…)",
                source_range: 74..75,
                delete: 74..75,
                insert: "foo(${1:x})$0",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(&self, x: i32)",
                trigger_call_info: true,
            },
        ]
        "###
        )
    }

    #[test]
    fn no_arg_snippets_for_method_call() {
        assert_debug_snapshot!(
            do_reference_completion_with_options(
                r"
                struct S {}
                impl S {
                    fn foo(&self, x: i32) {}
                }
                fn bar(s: &S) {
                    s.f<|>
                }
                ",
                CompletionConfig {
                    add_call_argument_snippets: false,
                    .. Default::default()
                }
            ),
            @r###"
        [
            CompletionItem {
                label: "foo(…)",
                source_range: 74..75,
                delete: 74..75,
                insert: "foo($0)",
                kind: Method,
                lookup: "foo",
                detail: "fn foo(&self, x: i32)",
                trigger_call_info: true,
            },
        ]
        "###
        )
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
                source_range: 40..41,
                delete: 40..41,
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
                source_range: 35..39,
                delete: 35..39,
                insert: "frobnicate",
                kind: Function,
                detail: "fn frobnicate()",
            },
            CompletionItem {
                label: "main",
                source_range: 35..39,
                delete: 35..39,
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
                source_range: 67..69,
                delete: 67..69,
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
        mark::check!(inserts_angle_brackets_for_generics);
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
                source_range: 28..30,
                delete: 28..30,
                insert: "Vec<$0>",
                kind: Struct,
                lookup: "Vec",
            },
            CompletionItem {
                label: "foo(…)",
                source_range: 28..30,
                delete: 28..30,
                insert: "foo(${1:xs})$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve)",
                trigger_call_info: true,
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
                source_range: 31..33,
                delete: 31..33,
                insert: "Vec<$0>",
                kind: TypeAlias,
                lookup: "Vec",
            },
            CompletionItem {
                label: "foo(…)",
                source_range: 31..33,
                delete: 31..33,
                insert: "foo(${1:xs})$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve)",
                trigger_call_info: true,
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
                source_range: 35..37,
                delete: 35..37,
                insert: "Vec",
                kind: Struct,
            },
            CompletionItem {
                label: "foo(…)",
                source_range: 35..37,
                delete: 35..37,
                insert: "foo(${1:xs})$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve)",
                trigger_call_info: true,
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
                source_range: 28..30,
                delete: 28..30,
                insert: "Vec",
                kind: Struct,
            },
            CompletionItem {
                label: "foo(…)",
                source_range: 28..30,
                delete: 28..30,
                insert: "foo(${1:xs})$0",
                kind: Function,
                lookup: "foo",
                detail: "fn foo(xs: Ve<i128>)",
                trigger_call_info: true,
            },
        ]
        "###
        );
    }

    #[test]
    fn dont_insert_macro_call_parens_unncessary() {
        mark::check!(dont_insert_macro_call_parens_unncessary);
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
                source_range: 9..9,
                delete: 9..9,
                insert: "frobnicate",
                kind: Macro,
                detail: "#[macro_export]\nmacro_rules! frobnicate",
            },
        ]
        "###
        );

        assert_debug_snapshot!(
            do_reference_completion(
                r"
                //- /main.rs
                macro_rules frobnicate {
                    () => ()
                }
                fn main() {
                    frob<|>!();
                }
                "
            ),
            @r###"
        [
            CompletionItem {
                label: "frobnicate!",
                source_range: 56..60,
                delete: 56..60,
                insert: "frobnicate",
                kind: Macro,
                detail: "macro_rules! frobnicate",
            },
            CompletionItem {
                label: "main()",
                source_range: 56..60,
                delete: 56..60,
                insert: "main()$0",
                kind: Function,
                lookup: "main",
                detail: "fn main()",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_in_func_call() {
        mark::check!(test_struct_field_completion_in_func_call);
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                struct A { another_field: i64, the_field: u32, my_string: String }
                fn test(my_param: u32) -> u32 { my_param }
                fn foo(a: A) {
                    test(a.<|>)
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "another_field",
                source_range: 136..136,
                delete: 136..136,
                insert: "another_field",
                kind: Field,
                detail: "i64",
            },
            CompletionItem {
                label: "my_string",
                source_range: 136..136,
                delete: 136..136,
                insert: "my_string",
                kind: Field,
                detail: "{unknown}",
            },
            CompletionItem {
                label: "the_field",
                source_range: 136..136,
                delete: 136..136,
                insert: "the_field",
                kind: Field,
                detail: "u32",
                score: TypeMatch,
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_in_func_call_with_type_and_name() {
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                struct A { another_field: i64, another_good_type: u32, the_field: u32 }
                fn test(the_field: u32) -> u32 { the_field }
                fn foo(a: A) {
                    test(a.<|>)
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "another_field",
                source_range: 143..143,
                delete: 143..143,
                insert: "another_field",
                kind: Field,
                detail: "i64",
            },
            CompletionItem {
                label: "another_good_type",
                source_range: 143..143,
                delete: 143..143,
                insert: "another_good_type",
                kind: Field,
                detail: "u32",
                score: TypeMatch,
            },
            CompletionItem {
                label: "the_field",
                source_range: 143..143,
                delete: 143..143,
                insert: "the_field",
                kind: Field,
                detail: "u32",
                score: TypeAndNameMatch,
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_in_record_lit() {
        mark::check!(test_struct_field_completion_in_record_lit);
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                struct A { another_field: i64, another_good_type: u32, the_field: u32 }
                struct B { my_string: String, my_vec: Vec<u32>, the_field: u32 }
                fn foo(a: A) {
                    let b = B {
                        the_field: a.<|>
                    };
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "another_field",
                source_range: 189..189,
                delete: 189..189,
                insert: "another_field",
                kind: Field,
                detail: "i64",
            },
            CompletionItem {
                label: "another_good_type",
                source_range: 189..189,
                delete: 189..189,
                insert: "another_good_type",
                kind: Field,
                detail: "u32",
                score: TypeMatch,
            },
            CompletionItem {
                label: "the_field",
                source_range: 189..189,
                delete: 189..189,
                insert: "the_field",
                kind: Field,
                detail: "u32",
                score: TypeAndNameMatch,
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_in_record_lit_and_fn_call() {
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                struct A { another_field: i64, another_good_type: u32, the_field: u32 }
                struct B { my_string: String, my_vec: Vec<u32>, the_field: u32 }
                fn test(the_field: i64) -> i64 { the_field }
                fn foo(a: A) {
                    let b = B {
                        the_field: test(a.<|>)
                    };
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "another_field",
                source_range: 239..239,
                delete: 239..239,
                insert: "another_field",
                kind: Field,
                detail: "i64",
                score: TypeMatch,
            },
            CompletionItem {
                label: "another_good_type",
                source_range: 239..239,
                delete: 239..239,
                insert: "another_good_type",
                kind: Field,
                detail: "u32",
            },
            CompletionItem {
                label: "the_field",
                source_range: 239..239,
                delete: 239..239,
                insert: "the_field",
                kind: Field,
                detail: "u32",
            },
        ]
        "###
        );
    }

    #[test]
    fn test_struct_field_completion_in_fn_call_and_record_lit() {
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                struct A { another_field: i64, another_good_type: u32, the_field: u32 }
                struct B { my_string: String, my_vec: Vec<u32>, the_field: u32 }
                fn test(the_field: i64) -> i64 { the_field }
                fn foo(a: A) {
                    test(B {
                        the_field: a.<|>
                    });
                }
                ",
        ),
            @r###"
        [
            CompletionItem {
                label: "another_field",
                source_range: 231..231,
                delete: 231..231,
                insert: "another_field",
                kind: Field,
                detail: "i64",
            },
            CompletionItem {
                label: "another_good_type",
                source_range: 231..231,
                delete: 231..231,
                insert: "another_good_type",
                kind: Field,
                detail: "u32",
                score: TypeMatch,
            },
            CompletionItem {
                label: "the_field",
                source_range: 231..231,
                delete: 231..231,
                insert: "the_field",
                kind: Field,
                detail: "u32",
                score: TypeAndNameMatch,
            },
        ]
        "###
        );
    }

    #[test]
    fn prioritize_exact_ref_match() {
        assert_debug_snapshot!(
        do_reference_completion(
                r"
                    struct WorldSnapshot { _f: () };
                    fn go(world: &WorldSnapshot) {
                        go(w<|>)
                    }
                    ",
        ),
            @r###"
        [
            CompletionItem {
                label: "WorldSnapshot",
                source_range: 71..72,
                delete: 71..72,
                insert: "WorldSnapshot",
                kind: Struct,
            },
            CompletionItem {
                label: "go(…)",
                source_range: 71..72,
                delete: 71..72,
                insert: "go(${1:world})$0",
                kind: Function,
                lookup: "go",
                detail: "fn go(world: &WorldSnapshot)",
                trigger_call_info: true,
            },
            CompletionItem {
                label: "world",
                source_range: 71..72,
                delete: 71..72,
                insert: "world",
                kind: Binding,
                detail: "&WorldSnapshot",
                score: TypeAndNameMatch,
            },
        ]
        "###
        );
    }
}
