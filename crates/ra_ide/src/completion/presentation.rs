//! This modules takes care of rendering various definitions as completion items.
//! It also handles scoring (sorting) completions.

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
                    .lookup_by(format!("{}!", name))
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
        mark::hit!(record_field_type_match);
        let (struct_field, _local) = ctx.sema.resolve_record_field(record_field)?;
        (
            struct_field.name(ctx.db).to_string(),
            struct_field.signature_ty(ctx.db).display(ctx.db).to_string(),
        )
    } else if let Some(active_parameter) = &ctx.active_parameter {
        mark::hit!(active_param_type_match);
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
            mark::hit!(no_parens_in_use_item);
            return self;
        }

        // Don't add parentheses if the expected type is some function reference.
        if let Some(ty) = &ctx.expected_type {
            if ty.is_fn() {
                mark::hit!(no_call_parens_if_fn_ptr_needed);
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
                _ => {
                    mark::hit!(suppress_arg_snippets);
                    format!("{}($0)", name)
                }
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
    use std::cmp::Reverse;

    use expect::{expect, Expect};
    use test_utils::mark;

    use crate::{
        completion::{
            test_utils::{
                check_edit, check_edit_with_config, do_completion, get_all_completion_items,
            },
            CompletionConfig, CompletionKind,
        },
        CompletionScore,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = do_completion(ra_fixture, CompletionKind::Reference);
        expect.assert_debug_eq(&actual);
    }

    fn check_scores(ra_fixture: &str, expect: Expect) {
        fn display_score(score: Option<CompletionScore>) -> &'static str {
            match score {
                Some(CompletionScore::TypeMatch) => "[type]",
                Some(CompletionScore::TypeAndNameMatch) => "[type+name]",
                None => "[]".into(),
            }
        }

        let mut completions = get_all_completion_items(ra_fixture, &CompletionConfig::default());
        completions.sort_by_key(|it| (Reverse(it.score()), it.label().to_string()));
        let actual = completions
            .into_iter()
            .filter(|it| it.completion_kind == CompletionKind::Reference)
            .map(|it| {
                let tag = it.kind().unwrap().tag();
                let score = display_score(it.score());
                format!("{} {} {}\n", tag, it.label(), score)
            })
            .collect::<String>();
        expect.assert_eq(&actual);
    }

    #[test]
    fn enum_detail_includes_record_fields() {
        check(
            r#"
enum Foo { Foo { x: i32, y: i32 } }

fn main() { Foo::Fo<|> }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Foo",
                        source_range: 54..56,
                        delete: 54..56,
                        insert: "Foo",
                        kind: EnumVariant,
                        detail: "{ x: i32, y: i32 }",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn enum_detail_doesnt_include_tuple_fields() {
        check(
            r#"
enum Foo { Foo (i32, i32) }

fn main() { Foo::Fo<|> }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Foo(…)",
                        source_range: 46..48,
                        delete: 46..48,
                        insert: "Foo($0)",
                        kind: EnumVariant,
                        lookup: "Foo",
                        detail: "(i32, i32)",
                        trigger_call_info: true,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn enum_detail_just_parentheses_for_unit() {
        check(
            r#"
enum Foo { Foo }

fn main() { Foo::Fo<|> }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Foo",
                        source_range: 35..37,
                        delete: 35..37,
                        insert: "Foo",
                        kind: EnumVariant,
                        detail: "()",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn sets_deprecated_flag_in_completion_items() {
        check(
            r#"
#[deprecated]
fn something_deprecated() {}
#[deprecated(since = "1.0.0")]
fn something_else_deprecated() {}

fn main() { som<|> }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "main()",
                        source_range: 121..124,
                        delete: 121..124,
                        insert: "main()$0",
                        kind: Function,
                        lookup: "main",
                        detail: "fn main()",
                    },
                    CompletionItem {
                        label: "something_deprecated()",
                        source_range: 121..124,
                        delete: 121..124,
                        insert: "something_deprecated()$0",
                        kind: Function,
                        lookup: "something_deprecated",
                        detail: "fn something_deprecated()",
                        deprecated: true,
                    },
                    CompletionItem {
                        label: "something_else_deprecated()",
                        source_range: 121..124,
                        delete: 121..124,
                        insert: "something_else_deprecated()$0",
                        kind: Function,
                        lookup: "something_else_deprecated",
                        detail: "fn something_else_deprecated()",
                        deprecated: true,
                    },
                ]
            "#]],
        );

        check(
            r#"
struct A { #[deprecated] the_field: u32 }
fn foo() { A { the<|> } }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "the_field",
                        source_range: 57..60,
                        delete: 57..60,
                        insert: "the_field",
                        kind: Field,
                        detail: "u32",
                        deprecated: true,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn renders_docs() {
        check(
            r#"
struct S {
    /// Field docs
    foo:
}
impl S {
    /// Method docs
    fn bar(self) { self.<|> }
}"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "bar()",
                        source_range: 94..94,
                        delete: 94..94,
                        insert: "bar()$0",
                        kind: Method,
                        lookup: "bar",
                        detail: "fn bar(self)",
                        documentation: Documentation(
                            "Method docs",
                        ),
                    },
                    CompletionItem {
                        label: "foo",
                        source_range: 94..94,
                        delete: 94..94,
                        insert: "foo",
                        kind: Field,
                        detail: "{unknown}",
                        documentation: Documentation(
                            "Field docs",
                        ),
                    },
                ]
            "#]],
        );

        check(
            r#"
use self::my<|>;

/// mod docs
mod my { }

/// enum docs
enum E {
    /// variant docs
    V
}
use self::E::*;
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "E",
                        source_range: 10..12,
                        delete: 10..12,
                        insert: "E",
                        kind: Enum,
                        documentation: Documentation(
                            "enum docs",
                        ),
                    },
                    CompletionItem {
                        label: "V",
                        source_range: 10..12,
                        delete: 10..12,
                        insert: "V",
                        kind: EnumVariant,
                        detail: "()",
                        documentation: Documentation(
                            "variant docs",
                        ),
                    },
                    CompletionItem {
                        label: "my",
                        source_range: 10..12,
                        delete: 10..12,
                        insert: "my",
                        kind: Module,
                        documentation: Documentation(
                            "mod docs",
                        ),
                    },
                ]
            "#]],
        )
    }

    #[test]
    fn dont_render_attrs() {
        check(
            r#"
struct S;
impl S {
    #[inline]
    fn the_method(&self) { }
}
fn foo(s: S) { s.<|> }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "the_method()",
                        source_range: 81..81,
                        delete: 81..81,
                        insert: "the_method()$0",
                        kind: Method,
                        lookup: "the_method",
                        detail: "fn the_method(&self)",
                    },
                ]
            "#]],
        )
    }

    #[test]
    fn inserts_parens_for_function_calls() {
        mark::check!(inserts_parens_for_function_calls);
        check_edit(
            "no_args",
            r#"
fn no_args() {}
fn main() { no_<|> }
"#,
            r#"
fn no_args() {}
fn main() { no_args()$0 }
"#,
        );

        check_edit(
            "with_args",
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_<|> }
"#,
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_args(${1:x}, ${2:y})$0 }
"#,
        );

        check_edit(
            "foo",
            r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn bar(s: &S) { s.f<|> }
"#,
            r#"
struct S;
impl S {
    fn foo(&self) {}
}
fn bar(s: &S) { s.foo()$0 }
"#,
        );

        check_edit(
            "foo",
            r#"
struct S {}
impl S {
    fn foo(&self, x: i32) {}
}
fn bar(s: &S) {
    s.f<|>
}
"#,
            r#"
struct S {}
impl S {
    fn foo(&self, x: i32) {}
}
fn bar(s: &S) {
    s.foo(${1:x})$0
}
"#,
        );
    }

    #[test]
    fn suppress_arg_snippets() {
        mark::check!(suppress_arg_snippets);
        check_edit_with_config(
            "with_args",
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_<|> }
"#,
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_args($0) }
"#,
            &CompletionConfig { add_call_argument_snippets: false, ..CompletionConfig::default() },
        );
    }

    #[test]
    fn strips_underscores_from_args() {
        check_edit(
            "foo",
            r#"
fn foo(_foo: i32, ___bar: bool, ho_ge_: String) {}
fn main() { f<|> }
"#,
            r#"
fn foo(_foo: i32, ___bar: bool, ho_ge_: String) {}
fn main() { foo(${1:foo}, ${2:bar}, ${3:ho_ge_})$0 }
"#,
        );
    }

    #[test]
    fn inserts_parens_for_tuple_enums() {
        check_edit(
            "Some",
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main() -> Option<i32> {
    Som<|>
}
"#,
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main() -> Option<i32> {
    Some($0)
}
"#,
        );
        check_edit(
            "Some",
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main(value: Option<i32>) {
    match value {
        Som<|>
    }
}
"#,
            r#"
enum Option<T> { Some(T), None }
use Option::*;
fn main(value: Option<i32>) {
    match value {
        Some($0)
    }
}
"#,
        );
    }

    #[test]
    fn no_call_parens_if_fn_ptr_needed() {
        mark::check!(no_call_parens_if_fn_ptr_needed);
        check_edit(
            "foo",
            r#"
fn foo(foo: u8, bar: u8) {}
struct ManualVtable { f: fn(u8, u8) }

fn main() -> ManualVtable {
    ManualVtable { f: f<|> }
}
"#,
            r#"
fn foo(foo: u8, bar: u8) {}
struct ManualVtable { f: fn(u8, u8) }

fn main() -> ManualVtable {
    ManualVtable { f: foo }
}
"#,
        );
    }

    #[test]
    fn no_parens_in_use_item() {
        mark::check!(no_parens_in_use_item);
        check_edit(
            "foo",
            r#"
mod m { pub fn foo() {} }
use crate::m::f<|>;
"#,
            r#"
mod m { pub fn foo() {} }
use crate::m::foo;
"#,
        );
    }

    #[test]
    fn no_parens_in_call() {
        check_edit(
            "foo",
            r#"
fn foo(x: i32) {}
fn main() { f<|>(); }
"#,
            r#"
fn foo(x: i32) {}
fn main() { foo(); }
"#,
        );
        check_edit(
            "foo",
            r#"
struct Foo;
impl Foo { fn foo(&self){} }
fn f(foo: &Foo) { foo.f<|>(); }
"#,
            r#"
struct Foo;
impl Foo { fn foo(&self){} }
fn f(foo: &Foo) { foo.foo(); }
"#,
        );
    }

    #[test]
    fn inserts_angle_brackets_for_generics() {
        mark::check!(inserts_angle_brackets_for_generics);
        check_edit(
            "Vec",
            r#"
struct Vec<T> {}
fn foo(xs: Ve<|>)
"#,
            r#"
struct Vec<T> {}
fn foo(xs: Vec<$0>)
"#,
        );
        check_edit(
            "Vec",
            r#"
type Vec<T> = (T,);
fn foo(xs: Ve<|>)
"#,
            r#"
type Vec<T> = (T,);
fn foo(xs: Vec<$0>)
"#,
        );
        check_edit(
            "Vec",
            r#"
struct Vec<T = i128> {}
fn foo(xs: Ve<|>)
"#,
            r#"
struct Vec<T = i128> {}
fn foo(xs: Vec)
"#,
        );
        check_edit(
            "Vec",
            r#"
struct Vec<T> {}
fn foo(xs: Ve<|><i128>)
"#,
            r#"
struct Vec<T> {}
fn foo(xs: Vec<i128>)
"#,
        );
    }

    #[test]
    fn dont_insert_macro_call_parens_unncessary() {
        mark::check!(dont_insert_macro_call_parens_unncessary);
        check_edit(
            "frobnicate!",
            r#"
//- /main.rs
use foo::<|>;
//- /foo/lib.rs
#[macro_export]
macro_rules frobnicate { () => () }
"#,
            r#"
use foo::frobnicate;
"#,
        );

        check_edit(
            "frobnicate!",
            r#"
macro_rules frobnicate { () => () }
fn main() { frob<|>!(); }
"#,
            r#"
macro_rules frobnicate { () => () }
fn main() { frobnicate!(); }
"#,
        );
    }

    #[test]
    fn active_param_score() {
        mark::check!(active_param_type_match);
        check_scores(
            r#"
struct S { foo: i64, bar: u32, baz: u32 }
fn test(bar: u32) { }
fn foo(s: S) { test(s.<|>) }
"#,
            expect![[r#"
                fd bar [type+name]
                fd baz [type]
                fd foo []
            "#]],
        );
    }

    #[test]
    fn record_field_scores() {
        mark::check!(record_field_type_match);
        check_scores(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn foo(a: A) { B { bar: a.<|> }; }
"#,
            expect![[r#"
                fd bar [type+name]
                fd baz [type]
                fd foo []
            "#]],
        )
    }

    #[test]
    fn record_field_and_call_scores() {
        check_scores(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn f(foo: i64) {  }
fn foo(a: A) { B { bar: f(a.<|>) }; }
"#,
            expect![[r#"
                fd foo [type+name]
                fd bar []
                fd baz []
            "#]],
        );
        check_scores(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn f(foo: i64) {  }
fn foo(a: A) { f(B { bar: a.<|> }); }
"#,
            expect![[r#"
                fd bar [type+name]
                fd baz [type]
                fd foo []
            "#]],
        );
    }

    #[test]
    fn prioritize_exact_ref_match() {
        check_scores(
            r#"
struct WorldSnapshot { _f: () };
fn go(world: &WorldSnapshot) { go(w<|>) }
"#,
            expect![[r#"
                bn world [type+name]
                st WorldSnapshot []
                fn go(…) []
            "#]],
        );
    }

    #[test]
    fn guesses_macro_braces() {
        check_edit(
            "vec!",
            r#"
/// Creates a [`Vec`] containing the arguments.
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
macro_rules! vec { () => {} }

fn fn main() { v<|> }
"#,
            r#"
/// Creates a [`Vec`] containing the arguments.
///
/// ```
/// let v = vec![1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// ```
macro_rules! vec { () => {} }

fn fn main() { vec![$0] }
"#,
        );

        check_edit(
            "foo!",
            r#"
/// Foo
///
/// Don't call `fooo!()` `fooo!()`, or `_foo![]` `_foo![]`,
/// call as `let _=foo!  { hello world };`
macro_rules! foo { () => {} }
fn main() { <|> }
"#,
            r#"
/// Foo
///
/// Don't call `fooo!()` `fooo!()`, or `_foo![]` `_foo![]`,
/// call as `let _=foo!  { hello world };`
macro_rules! foo { () => {} }
fn main() { foo! {$0} }
"#,
        )
    }
}
