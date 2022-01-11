//! `render` module provides utilities for rendering completion suggestions
//! into code pieces that will be presented to user.

pub(crate) mod macro_;
pub(crate) mod function;
pub(crate) mod enum_variant;
pub(crate) mod const_;
pub(crate) mod pattern;
pub(crate) mod type_alias;
pub(crate) mod struct_literal;

mod builder_ext;

use hir::{AsAssocItem, HasAttrs, HirDisplay, ScopeDef};
use ide_db::{
    helpers::{item_name, SnippetCap},
    RootDatabase, SymbolKind,
};
use syntax::{SmolStr, SyntaxKind, TextRange};

use crate::{
    context::{PathCompletionContext, PathKind},
    item::{CompletionRelevanceTypeMatch, ImportEdit},
    render::{enum_variant::render_variant, function::render_fn, macro_::render_macro},
    CompletionContext, CompletionItem, CompletionItemKind, CompletionRelevance,
};
/// Interface for data and methods required for items rendering.
#[derive(Debug)]
pub(crate) struct RenderContext<'a> {
    completion: &'a CompletionContext<'a>,
}

impl<'a> RenderContext<'a> {
    pub(crate) fn new(completion: &'a CompletionContext<'a>) -> RenderContext<'a> {
        RenderContext { completion }
    }

    fn snippet_cap(&self) -> Option<SnippetCap> {
        self.completion.config.snippet_cap
    }

    fn db(&self) -> &'a RootDatabase {
        self.completion.db
    }

    fn source_range(&self) -> TextRange {
        self.completion.source_range()
    }

    fn is_deprecated(&self, def: impl HasAttrs) -> bool {
        let attrs = def.attrs(self.db());
        attrs.by_key("deprecated").exists() || attrs.by_key("rustc_deprecated").exists()
    }

    fn is_deprecated_assoc_item(&self, as_assoc_item: impl AsAssocItem) -> bool {
        let db = self.db();
        let assoc = match as_assoc_item.as_assoc_item(db) {
            Some(assoc) => assoc,
            None => return false,
        };

        let is_assoc_deprecated = match assoc {
            hir::AssocItem::Function(it) => self.is_deprecated(it),
            hir::AssocItem::Const(it) => self.is_deprecated(it),
            hir::AssocItem::TypeAlias(it) => self.is_deprecated(it),
        };
        is_assoc_deprecated
            || assoc
                .containing_trait_or_trait_impl(db)
                .map(|trait_| self.is_deprecated(trait_))
                .unwrap_or(false)
    }

    // FIXME: remove this
    fn docs(&self, def: impl HasAttrs) -> Option<hir::Documentation> {
        def.docs(self.db())
    }
}

pub(crate) fn render_field(
    ctx: RenderContext<'_>,
    receiver: Option<hir::Name>,
    field: hir::Field,
    ty: &hir::Type,
) -> CompletionItem {
    let is_deprecated = ctx.is_deprecated(field);
    let name = field.name(ctx.db()).to_smol_str();
    let mut item = CompletionItem::new(
        SymbolKind::Field,
        ctx.source_range(),
        receiver.map_or_else(|| name.clone(), |receiver| format!("{}.{}", receiver, name).into()),
    );
    item.set_relevance(CompletionRelevance {
        type_match: compute_type_match(ctx.completion, ty),
        exact_name_match: compute_exact_name_match(ctx.completion, name.as_str()),
        ..CompletionRelevance::default()
    });
    item.detail(ty.display(ctx.db()).to_string())
        .set_documentation(field.docs(ctx.db()))
        .set_deprecated(is_deprecated)
        .lookup_by(name.clone());
    let is_keyword = SyntaxKind::from_keyword(name.as_str()).is_some();
    if is_keyword && !matches!(name.as_str(), "self" | "crate" | "super" | "Self") {
        item.insert_text(format!("r#{}", name));
    }
    if let Some(_ref_match) = compute_ref_match(ctx.completion, ty) {
        // FIXME
        // For now we don't properly calculate the edits for ref match
        // completions on struct fields, so we've disabled them. See #8058.
    }
    item.build()
}

pub(crate) fn render_tuple_field(
    ctx: RenderContext<'_>,
    receiver: Option<hir::Name>,
    field: usize,
    ty: &hir::Type,
) -> CompletionItem {
    let mut item = CompletionItem::new(
        SymbolKind::Field,
        ctx.source_range(),
        receiver.map_or_else(|| field.to_string(), |receiver| format!("{}.{}", receiver, field)),
    );
    item.detail(ty.display(ctx.db()).to_string()).lookup_by(field.to_string());
    item.build()
}

pub(crate) fn render_resolution(
    ctx: RenderContext<'_>,
    local_name: hir::Name,
    resolution: ScopeDef,
) -> CompletionItem {
    render_resolution_(ctx, local_name, None, resolution)
}

pub(crate) fn render_resolution_with_import(
    ctx: RenderContext<'_>,
    import_edit: ImportEdit,
) -> Option<CompletionItem> {
    let resolution = ScopeDef::from(import_edit.import.original_item);
    let local_name = match resolution {
        ScopeDef::ModuleDef(hir::ModuleDef::Function(f)) => f.name(ctx.completion.db),
        ScopeDef::ModuleDef(hir::ModuleDef::Const(c)) => c.name(ctx.completion.db)?,
        ScopeDef::ModuleDef(hir::ModuleDef::TypeAlias(t)) => t.name(ctx.completion.db),
        _ => item_name(ctx.db(), import_edit.import.original_item)?,
    };
    Some(render_resolution_(ctx, local_name, Some(import_edit), resolution))
}

fn render_resolution_(
    ctx: RenderContext<'_>,
    local_name: hir::Name,
    import_to_add: Option<ImportEdit>,
    resolution: ScopeDef,
) -> CompletionItem {
    let _p = profile::span("render_resolution");
    use hir::ModuleDef::*;

    let db = ctx.db();

    let kind = match resolution {
        ScopeDef::ModuleDef(Function(func)) => {
            return render_fn(ctx, import_to_add, Some(local_name), func);
        }
        ScopeDef::ModuleDef(Variant(var)) if ctx.completion.pattern_ctx.is_none() => {
            return render_variant(ctx, import_to_add, Some(local_name), var, None);
        }
        ScopeDef::MacroDef(mac) => return render_macro(ctx, import_to_add, local_name, mac),
        ScopeDef::Unknown => {
            let mut item = CompletionItem::new(
                CompletionItemKind::UnresolvedReference,
                ctx.source_range(),
                local_name.to_smol_str(),
            );
            if let Some(import_to_add) = import_to_add {
                item.add_import(import_to_add);
            }
            return item.build();
        }

        ScopeDef::ModuleDef(Variant(_)) => CompletionItemKind::SymbolKind(SymbolKind::Variant),
        ScopeDef::ModuleDef(Module(..)) => CompletionItemKind::SymbolKind(SymbolKind::Module),
        ScopeDef::ModuleDef(Adt(adt)) => CompletionItemKind::SymbolKind(match adt {
            hir::Adt::Struct(_) => SymbolKind::Struct,
            hir::Adt::Union(_) => SymbolKind::Union,
            hir::Adt::Enum(_) => SymbolKind::Enum,
        }),
        ScopeDef::ModuleDef(Const(..)) => CompletionItemKind::SymbolKind(SymbolKind::Const),
        ScopeDef::ModuleDef(Static(..)) => CompletionItemKind::SymbolKind(SymbolKind::Static),
        ScopeDef::ModuleDef(Trait(..)) => CompletionItemKind::SymbolKind(SymbolKind::Trait),
        ScopeDef::ModuleDef(TypeAlias(..)) => CompletionItemKind::SymbolKind(SymbolKind::TypeAlias),
        ScopeDef::ModuleDef(BuiltinType(..)) => CompletionItemKind::BuiltinType,
        ScopeDef::GenericParam(param) => CompletionItemKind::SymbolKind(match param {
            hir::GenericParam::TypeParam(_) => SymbolKind::TypeParam,
            hir::GenericParam::LifetimeParam(_) => SymbolKind::LifetimeParam,
            hir::GenericParam::ConstParam(_) => SymbolKind::ConstParam,
        }),
        ScopeDef::Local(..) => CompletionItemKind::SymbolKind(SymbolKind::Local),
        ScopeDef::Label(..) => CompletionItemKind::SymbolKind(SymbolKind::Label),
        ScopeDef::AdtSelfType(..) | ScopeDef::ImplSelfType(..) => {
            CompletionItemKind::SymbolKind(SymbolKind::SelfParam)
        }
    };

    let local_name = local_name.to_smol_str();
    let mut item = CompletionItem::new(kind, ctx.source_range(), local_name.clone());
    if let ScopeDef::Local(local) = resolution {
        let ty = local.ty(db);
        if !ty.is_unknown() {
            item.detail(ty.display(db).to_string());
        }

        item.set_relevance(CompletionRelevance {
            type_match: compute_type_match(ctx.completion, &ty),
            exact_name_match: compute_exact_name_match(ctx.completion, &local_name),
            is_local: true,
            ..CompletionRelevance::default()
        });

        if let Some(ref_match) = compute_ref_match(ctx.completion, &ty) {
            item.ref_match(ref_match);
        }
    };

    // Add `<>` for generic types
    let type_path_no_ty_args = matches!(
        ctx.completion.path_context,
        Some(PathCompletionContext { kind: Some(PathKind::Type), has_type_args: false, .. })
    ) && ctx.completion.config.add_call_parenthesis;
    if type_path_no_ty_args {
        if let Some(cap) = ctx.snippet_cap() {
            let has_non_default_type_params = match resolution {
                ScopeDef::ModuleDef(Adt(it)) => it.has_non_default_type_params(db),
                ScopeDef::ModuleDef(TypeAlias(it)) => it.has_non_default_type_params(db),
                _ => false,
            };
            if has_non_default_type_params {
                cov_mark::hit!(inserts_angle_brackets_for_generics);
                item.lookup_by(local_name.clone())
                    .label(SmolStr::from_iter([&local_name, "<…>"]))
                    .insert_snippet(cap, format!("{}<$0>", local_name));
            }
        }
    }
    item.set_documentation(scope_def_docs(db, resolution))
        .set_deprecated(scope_def_is_deprecated(&ctx, resolution));

    if let Some(import_to_add) = import_to_add {
        item.add_import(import_to_add);
    }
    item.build()
}

fn scope_def_docs(db: &RootDatabase, resolution: ScopeDef) -> Option<hir::Documentation> {
    use hir::ModuleDef::*;
    match resolution {
        ScopeDef::ModuleDef(Module(it)) => it.docs(db),
        ScopeDef::ModuleDef(Adt(it)) => it.docs(db),
        ScopeDef::ModuleDef(Variant(it)) => it.docs(db),
        ScopeDef::ModuleDef(Const(it)) => it.docs(db),
        ScopeDef::ModuleDef(Static(it)) => it.docs(db),
        ScopeDef::ModuleDef(Trait(it)) => it.docs(db),
        ScopeDef::ModuleDef(TypeAlias(it)) => it.docs(db),
        _ => None,
    }
}

fn scope_def_is_deprecated(ctx: &RenderContext<'_>, resolution: ScopeDef) -> bool {
    match resolution {
        ScopeDef::ModuleDef(it) => ctx.is_deprecated_assoc_item(it),
        ScopeDef::MacroDef(it) => ctx.is_deprecated(it),
        ScopeDef::GenericParam(it) => ctx.is_deprecated(it),
        ScopeDef::AdtSelfType(it) => ctx.is_deprecated(it),
        _ => false,
    }
}

fn compute_type_match(
    ctx: &CompletionContext,
    completion_ty: &hir::Type,
) -> Option<CompletionRelevanceTypeMatch> {
    let expected_type = ctx.expected_type.as_ref()?;

    // We don't ever consider unit type to be an exact type match, since
    // nearly always this is not meaningful to the user.
    if expected_type.is_unit() {
        return None;
    }

    if completion_ty == expected_type {
        Some(CompletionRelevanceTypeMatch::Exact)
    } else if expected_type.could_unify_with(ctx.db, completion_ty) {
        Some(CompletionRelevanceTypeMatch::CouldUnify)
    } else {
        None
    }
}

fn compute_exact_name_match(ctx: &CompletionContext, completion_name: &str) -> bool {
    ctx.expected_name.as_ref().map_or(false, |name| name.text() == completion_name)
}

fn compute_ref_match(
    ctx: &CompletionContext,
    completion_ty: &hir::Type,
) -> Option<hir::Mutability> {
    let expected_type = ctx.expected_type.as_ref()?;
    if completion_ty != expected_type {
        let expected_type_without_ref = expected_type.remove_ref()?;
        if completion_ty.autoderef(ctx.db).any(|deref_ty| deref_ty == expected_type_without_ref) {
            cov_mark::hit!(suggest_ref);
            let mutability = if expected_type.is_mutable_reference() {
                hir::Mutability::Mut
            } else {
                hir::Mutability::Shared
            };
            return Some(mutability);
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use expect_test::{expect, Expect};
    use ide_db::SymbolKind;
    use itertools::Itertools;

    use crate::{
        item::CompletionRelevanceTypeMatch,
        tests::{check_edit, do_completion, get_all_items, TEST_CONFIG},
        CompletionItem, CompletionItemKind, CompletionRelevance,
    };

    #[track_caller]
    fn check(ra_fixture: &str, kind: impl Into<CompletionItemKind>, expect: Expect) {
        let actual = do_completion(ra_fixture, kind.into());
        expect.assert_debug_eq(&actual);
    }

    #[track_caller]
    fn check_kinds(ra_fixture: &str, kinds: &[CompletionItemKind], expect: Expect) {
        let actual: Vec<_> =
            kinds.iter().flat_map(|&kind| do_completion(ra_fixture, kind)).collect();
        expect.assert_debug_eq(&actual);
    }

    #[track_caller]
    fn check_relevance_for_kinds(ra_fixture: &str, kinds: &[CompletionItemKind], expect: Expect) {
        let mut actual = get_all_items(TEST_CONFIG, ra_fixture);
        actual.retain(|it| kinds.contains(&it.kind()));
        actual.sort_by_key(|it| cmp::Reverse(it.relevance().score()));
        check_relevance_(actual, expect);
    }

    #[track_caller]
    fn check_relevance(ra_fixture: &str, expect: Expect) {
        let mut actual = get_all_items(TEST_CONFIG, ra_fixture);
        actual.retain(|it| it.kind() != CompletionItemKind::Snippet);
        actual.retain(|it| it.kind() != CompletionItemKind::Keyword);
        actual.retain(|it| it.kind() != CompletionItemKind::BuiltinType);
        actual.sort_by_key(|it| cmp::Reverse(it.relevance().score()));
        check_relevance_(actual, expect);
    }

    #[track_caller]
    fn check_relevance_(actual: Vec<CompletionItem>, expect: Expect) {
        let actual = actual
            .into_iter()
            .flat_map(|it| {
                let mut items = vec![];

                let tag = it.kind().tag();
                let relevance = display_relevance(it.relevance());
                items.push(format!("{} {} {}\n", tag, it.label(), relevance));

                if let Some((mutability, relevance)) = it.ref_match() {
                    let label = format!("&{}{}", mutability.as_keyword_for_ref(), it.label());
                    let relevance = display_relevance(relevance);

                    items.push(format!("{} {} {}\n", tag, label, relevance));
                }

                items
            })
            .collect::<String>();

        expect.assert_eq(&actual);

        fn display_relevance(relevance: CompletionRelevance) -> String {
            let relevance_factors = vec![
                (relevance.type_match == Some(CompletionRelevanceTypeMatch::Exact), "type"),
                (
                    relevance.type_match == Some(CompletionRelevanceTypeMatch::CouldUnify),
                    "type_could_unify",
                ),
                (relevance.exact_name_match, "name"),
                (relevance.is_local, "local"),
                (relevance.exact_postfix_snippet_match, "snippet"),
                (relevance.is_op_method, "op_method"),
            ]
            .into_iter()
            .filter_map(|(cond, desc)| if cond { Some(desc) } else { None })
            .join("+");

            format!("[{}]", relevance_factors)
        }
    }

    #[test]
    fn enum_detail_includes_record_fields() {
        check(
            r#"
enum Foo { Foo { x: i32, y: i32 } }

fn main() { Foo::Fo$0 }
"#,
            SymbolKind::Variant,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Foo",
                        source_range: 54..56,
                        delete: 54..56,
                        insert: "Foo",
                        kind: SymbolKind(
                            Variant,
                        ),
                        detail: "{x: i32, y: i32}",
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

fn main() { Foo::Fo$0 }
"#,
            SymbolKind::Variant,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Foo(…)",
                        source_range: 46..48,
                        delete: 46..48,
                        insert: "Foo($0)",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Foo",
                        detail: "(i32, i32)",
                        trigger_call_info: true,
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn fn_detail_includes_args_and_return_type() {
        check(
            r#"
fn foo<T>(a: u32, b: u32, t: T) -> (u32, T) { (a, t) }

fn main() { fo$0 }
"#,
            SymbolKind::Function,
            expect![[r#"
                [
                    CompletionItem {
                        label: "foo(…)",
                        source_range: 68..70,
                        delete: 68..70,
                        insert: "foo(${1:a}, ${2:b}, ${3:t})$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "foo",
                        detail: "fn(u32, u32, T) -> (u32, T)",
                        trigger_call_info: true,
                    },
                    CompletionItem {
                        label: "main()",
                        source_range: 68..70,
                        delete: 68..70,
                        insert: "main()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "fn()",
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

fn main() { Foo::Fo$0 }
"#,
            SymbolKind::Variant,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Foo",
                        source_range: 35..37,
                        delete: 35..37,
                        insert: "Foo",
                        kind: SymbolKind(
                            Variant,
                        ),
                        detail: "()",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn lookup_enums_by_two_qualifiers() {
        check_kinds(
            r#"
mod m {
    pub enum Spam { Foo, Bar(i32) }
}
fn main() { let _: m::Spam = S$0 }
"#,
            &[
                CompletionItemKind::SymbolKind(SymbolKind::Function),
                CompletionItemKind::SymbolKind(SymbolKind::Module),
                CompletionItemKind::SymbolKind(SymbolKind::Variant),
            ],
            expect![[r#"
                [
                    CompletionItem {
                        label: "main()",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "main()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "fn()",
                    },
                    CompletionItem {
                        label: "m",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m",
                        kind: SymbolKind(
                            Module,
                        ),
                    },
                    CompletionItem {
                        label: "Spam::Bar(…)",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "Spam::Bar($0)",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Bar",
                        detail: "(i32)",
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                Exact,
                            ),
                            is_local: false,
                            is_op_method: false,
                            exact_postfix_snippet_match: false,
                        },
                        trigger_call_info: true,
                    },
                    CompletionItem {
                        label: "m::Spam::Foo",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Foo",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Foo",
                        detail: "()",
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                Exact,
                            ),
                            is_local: false,
                            is_op_method: false,
                            exact_postfix_snippet_match: false,
                        },
                    },
                ]
            "#]],
        )
    }

    #[test]
    fn sets_deprecated_flag_in_items() {
        check(
            r#"
#[deprecated]
fn something_deprecated() {}
#[rustc_deprecated(since = "1.0.0")]
fn something_else_deprecated() {}

fn main() { som$0 }
"#,
            SymbolKind::Function,
            expect![[r#"
                [
                    CompletionItem {
                        label: "main()",
                        source_range: 127..130,
                        delete: 127..130,
                        insert: "main()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "fn()",
                    },
                    CompletionItem {
                        label: "something_deprecated()",
                        source_range: 127..130,
                        delete: 127..130,
                        insert: "something_deprecated()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "something_deprecated",
                        detail: "fn()",
                        deprecated: true,
                    },
                    CompletionItem {
                        label: "something_else_deprecated()",
                        source_range: 127..130,
                        delete: 127..130,
                        insert: "something_else_deprecated()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "something_else_deprecated",
                        detail: "fn()",
                        deprecated: true,
                    },
                ]
            "#]],
        );

        check(
            r#"
struct A { #[deprecated] the_field: u32 }
fn foo() { A { the$0 } }
"#,
            SymbolKind::Field,
            expect![[r#"
                [
                    CompletionItem {
                        label: "the_field",
                        source_range: 57..60,
                        delete: 57..60,
                        insert: "the_field",
                        kind: SymbolKind(
                            Field,
                        ),
                        detail: "u32",
                        deprecated: true,
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                CouldUnify,
                            ),
                            is_local: false,
                            is_op_method: false,
                            exact_postfix_snippet_match: false,
                        },
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn renders_docs() {
        check_kinds(
            r#"
struct S {
    /// Field docs
    foo:
}
impl S {
    /// Method docs
    fn bar(self) { self.$0 }
}"#,
            &[CompletionItemKind::Method, CompletionItemKind::SymbolKind(SymbolKind::Field)],
            expect![[r#"
                [
                    CompletionItem {
                        label: "bar()",
                        source_range: 94..94,
                        delete: 94..94,
                        insert: "bar()$0",
                        kind: Method,
                        lookup: "bar",
                        detail: "fn(self)",
                        documentation: Documentation(
                            "Method docs",
                        ),
                    },
                    CompletionItem {
                        label: "foo",
                        source_range: 94..94,
                        delete: 94..94,
                        insert: "foo",
                        kind: SymbolKind(
                            Field,
                        ),
                        detail: "{unknown}",
                        documentation: Documentation(
                            "Field docs",
                        ),
                    },
                ]
            "#]],
        );

        check_kinds(
            r#"
use self::my$0;

/// mod docs
mod my { }

/// enum docs
enum E {
    /// variant docs
    V
}
use self::E::*;
"#,
            &[
                CompletionItemKind::SymbolKind(SymbolKind::Module),
                CompletionItemKind::SymbolKind(SymbolKind::Variant),
                CompletionItemKind::SymbolKind(SymbolKind::Enum),
            ],
            expect![[r#"
                [
                    CompletionItem {
                        label: "my",
                        source_range: 10..12,
                        delete: 10..12,
                        insert: "my",
                        kind: SymbolKind(
                            Module,
                        ),
                        documentation: Documentation(
                            "mod docs",
                        ),
                    },
                    CompletionItem {
                        label: "V",
                        source_range: 10..12,
                        delete: 10..12,
                        insert: "V",
                        kind: SymbolKind(
                            Variant,
                        ),
                        detail: "()",
                        documentation: Documentation(
                            "variant docs",
                        ),
                    },
                    CompletionItem {
                        label: "E",
                        source_range: 10..12,
                        delete: 10..12,
                        insert: "E",
                        kind: SymbolKind(
                            Enum,
                        ),
                        documentation: Documentation(
                            "enum docs",
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
fn foo(s: S) { s.$0 }
"#,
            CompletionItemKind::Method,
            expect![[r#"
                [
                    CompletionItem {
                        label: "the_method()",
                        source_range: 81..81,
                        delete: 81..81,
                        insert: "the_method()$0",
                        kind: Method,
                        lookup: "the_method",
                        detail: "fn(&self)",
                    },
                ]
            "#]],
        )
    }

    #[test]
    fn no_call_parens_if_fn_ptr_needed() {
        cov_mark::check!(no_call_parens_if_fn_ptr_needed);
        check_edit(
            "foo",
            r#"
fn foo(foo: u8, bar: u8) {}
struct ManualVtable { f: fn(u8, u8) }

fn main() -> ManualVtable {
    ManualVtable { f: f$0 }
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
        check_edit(
            "type",
            r#"
struct RawIdentTable { r#type: u32 }

fn main() -> RawIdentTable {
    RawIdentTable { t$0: 42 }
}
"#,
            r#"
struct RawIdentTable { r#type: u32 }

fn main() -> RawIdentTable {
    RawIdentTable { r#type: 42 }
}
"#,
        );
    }

    #[test]
    fn no_parens_in_use_item() {
        cov_mark::check!(no_parens_in_use_item);
        check_edit(
            "foo",
            r#"
mod m { pub fn foo() {} }
use crate::m::f$0;
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
fn main() { f$0(); }
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
fn f(foo: &Foo) { foo.f$0(); }
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
        cov_mark::check!(inserts_angle_brackets_for_generics);
        check_edit(
            "Vec",
            r#"
struct Vec<T> {}
fn foo(xs: Ve$0)
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
fn foo(xs: Ve$0)
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
fn foo(xs: Ve$0)
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
fn foo(xs: Ve$0<i128>)
"#,
            r#"
struct Vec<T> {}
fn foo(xs: Vec<i128>)
"#,
        );
    }

    #[test]
    fn active_param_relevance() {
        check_relevance(
            r#"
struct S { foo: i64, bar: u32, baz: u32 }
fn test(bar: u32) { }
fn foo(s: S) { test(s.$0) }
"#,
            expect![[r#"
                fd bar [type+name]
                fd baz [type]
                fd foo []
            "#]],
        );
    }

    #[test]
    fn record_field_relevances() {
        check_relevance(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn foo(a: A) { B { bar: a.$0 }; }
"#,
            expect![[r#"
                fd bar [type+name]
                fd baz [type]
                fd foo []
            "#]],
        )
    }

    #[test]
    fn record_field_and_call_relevances() {
        check_relevance(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn f(foo: i64) {  }
fn foo(a: A) { B { bar: f(a.$0) }; }
"#,
            expect![[r#"
                fd foo [type+name]
                fd bar []
                fd baz []
            "#]],
        );
        check_relevance(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn f(foo: i64) {  }
fn foo(a: A) { f(B { bar: a.$0 }); }
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
        check_relevance(
            r#"
struct WorldSnapshot { _f: () };
fn go(world: &WorldSnapshot) { go(w$0) }
"#,
            expect![[r#"
                lc world [type+name+local]
                st WorldSnapshot []
                fn go(…) []
            "#]],
        );
    }

    #[test]
    fn too_many_arguments() {
        cov_mark::check!(too_many_arguments);
        check_relevance(
            r#"
struct Foo;
fn f(foo: &Foo) { f(foo, w$0) }
"#,
            expect![[r#"
                lc foo [local]
                st Foo []
                fn f(…) []
            "#]],
        );
    }

    #[test]
    fn score_fn_type_and_name_match() {
        check_relevance(
            r#"
struct A { bar: u8 }
fn baz() -> u8 { 0 }
fn bar() -> u8 { 0 }
fn f() { A { bar: b$0 }; }
"#,
            expect![[r#"
                fn bar() [type+name]
                fn baz() [type]
                st A []
                fn f() []
            "#]],
        );
    }

    #[test]
    fn score_method_type_and_name_match() {
        check_relevance(
            r#"
fn baz(aaa: u32){}
struct Foo;
impl Foo {
fn aaa(&self) -> u32 { 0 }
fn bbb(&self) -> u32 { 0 }
fn ccc(&self) -> u64 { 0 }
}
fn f() {
    baz(Foo.$0
}
"#,
            expect![[r#"
                me aaa() [type+name]
                me bbb() [type]
                me ccc() []
            "#]],
        );
    }

    #[test]
    fn score_method_name_match_only() {
        check_relevance(
            r#"
fn baz(aaa: u32){}
struct Foo;
impl Foo {
fn aaa(&self) -> u64 { 0 }
}
fn f() {
    baz(Foo.$0
}
"#,
            expect![[r#"
                me aaa() [name]
            "#]],
        );
    }

    #[test]
    fn suggest_ref_mut() {
        cov_mark::check!(suggest_ref);
        check_relevance(
            r#"
struct S;
fn foo(s: &mut S) {}
fn main() {
    let mut s = S;
    foo($0);
}
            "#,
            expect![[r#"
                lc s [name+local]
                lc &mut s [type+name+local]
                st S []
                fn main() []
                fn foo(…) []
            "#]],
        );
        check_relevance(
            r#"
struct S;
fn foo(s: &mut S) {}
fn main() {
    let mut s = S;
    foo(&mut $0);
}
            "#,
            expect![[r#"
                lc s [type+name+local]
                st S []
                fn main() []
                fn foo(…) []
            "#]],
        );
        check_relevance(
            r#"
struct S;
fn foo(s: &mut S) {}
fn main() {
    let mut ssss = S;
    foo(&mut s$0);
}
            "#,
            expect![[r#"
                lc ssss [type+local]
                st S []
                fn main() []
                fn foo(…) []
            "#]],
        );
    }

    #[test]
    fn suggest_deref() {
        check_relevance(
            r#"
//- minicore: deref
struct S;
struct T(S);

impl core::ops::Deref for T {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn foo(s: &S) {}

fn main() {
    let t = T(S);
    let m = 123;

    foo($0);
}
            "#,
            expect![[r#"
                lc m [local]
                lc t [local]
                lc &t [type+local]
                st T []
                st S []
                fn main() []
                fn foo(…) []
                md core []
                tt Sized []
            "#]],
        )
    }

    #[test]
    fn suggest_deref_mut() {
        check_relevance(
            r#"
//- minicore: deref_mut
struct S;
struct T(S);

impl core::ops::Deref for T {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::DerefMut for T {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn foo(s: &mut S) {}

fn main() {
    let t = T(S);
    let m = 123;

    foo($0);
}
            "#,
            expect![[r#"
                lc m [local]
                lc t [local]
                lc &mut t [type+local]
                st T []
                st S []
                fn main() []
                fn foo(…) []
                md core []
                tt Sized []
            "#]],
        )
    }

    #[test]
    fn locals() {
        check_relevance(
            r#"
fn foo(bar: u32) {
    let baz = 0;

    f$0
}
"#,
            expect![[r#"
                lc baz [local]
                lc bar [local]
                fn foo(…) []
            "#]],
        );
    }

    #[test]
    fn enum_owned() {
        check_relevance(
            r#"
enum Foo { A, B }
fn foo() {
    bar($0);
}
fn bar(t: Foo) {}
"#,
            expect![[r#"
                ev Foo::A [type]
                ev Foo::B [type]
                en Foo []
                fn bar(…) []
                fn foo() []
            "#]],
        );
    }

    #[test]
    fn enum_ref() {
        check_relevance(
            r#"
enum Foo { A, B }
fn foo() {
    bar($0);
}
fn bar(t: &Foo) {}
"#,
            expect![[r#"
                ev Foo::A []
                ev &Foo::A [type]
                ev Foo::B []
                ev &Foo::B [type]
                en Foo []
                fn bar(…) []
                fn foo() []
            "#]],
        );
    }

    #[test]
    fn suggest_deref_fn_ret() {
        check_relevance(
            r#"
//- minicore: deref
struct S;
struct T(S);

impl core::ops::Deref for T {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn foo(s: &S) {}
fn bar() -> T {}

fn main() {
    foo($0);
}
"#,
            expect![[r#"
                st T []
                st S []
                fn main() []
                fn bar() []
                fn &bar() [type]
                fn foo(…) []
                md core []
                tt Sized []
            "#]],
        )
    }

    #[test]
    fn op_method_relevances() {
        check_relevance(
            r#"
#[lang = "sub"]
trait Sub {
    fn sub(self, other: Self) -> Self { self }
}
impl Sub for u32 {}
fn foo(a: u32) { a.$0 }
"#,
            expect![[r#"
                me sub(…) (as Sub) [op_method]
            "#]],
        )
    }

    #[test]
    fn struct_field_method_ref() {
        check_kinds(
            r#"
struct Foo { bar: u32 }
impl Foo { fn baz(&self) -> u32 { 0 } }

fn foo(f: Foo) { let _: &u32 = f.b$0 }
"#,
            &[CompletionItemKind::Method, CompletionItemKind::SymbolKind(SymbolKind::Field)],
            // FIXME
            // Ideally we'd also suggest &f.bar and &f.baz() as exact
            // type matches. See #8058.
            expect![[r#"
                [
                    CompletionItem {
                        label: "baz()",
                        source_range: 98..99,
                        delete: 98..99,
                        insert: "baz()$0",
                        kind: Method,
                        lookup: "baz",
                        detail: "fn(&self) -> u32",
                    },
                    CompletionItem {
                        label: "bar",
                        source_range: 98..99,
                        delete: 98..99,
                        insert: "bar",
                        kind: SymbolKind(
                            Field,
                        ),
                        detail: "u32",
                    },
                ]
            "#]],
        );
    }

    #[test]
    fn generic_enum() {
        check_relevance(
            r#"
enum Foo<T> { A(T), B }
// bar() should not be an exact type match
// because the generic parameters are different
fn bar() -> Foo<u8> { Foo::B }
// FIXME baz() should be an exact type match
// because the types could unify, but it currently
// is not. This is due to the T here being
// TyKind::Placeholder rather than TyKind::Missing.
fn baz<T>() -> Foo<T> { Foo::B }
fn foo() {
    let foo: Foo<u32> = Foo::B;
    let _: Foo<u32> = f$0;
}
"#,
            expect![[r#"
                lc foo [type+local]
                ev Foo::A(…) [type_could_unify]
                ev Foo::B [type_could_unify]
                fn foo() []
                en Foo []
                fn baz() []
                fn bar() []
            "#]],
        );
    }

    #[test]
    fn postfix_completion_relevance() {
        check_relevance_for_kinds(
            r#"
mod ops {
    pub trait Not {
        type Output;
        fn not(self) -> Self::Output;
    }

    impl Not for bool {
        type Output = bool;
        fn not(self) -> bool { if self { false } else { true }}
    }
}

fn main() {
    let _: bool = (9 > 2).not$0;
}
    "#,
            &[CompletionItemKind::Snippet, CompletionItemKind::Method],
            expect![[r#"
                sn not [snippet]
                me not() (use ops::Not) [type_could_unify]
                sn if []
                sn while []
                sn ref []
                sn refm []
                sn match []
                sn box []
                sn dbg []
                sn dbgr []
                sn call []
            "#]],
        );
    }
}
