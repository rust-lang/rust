//! `render` module provides utilities for rendering completion suggestions
//! into code pieces that will be presented to user.

pub(crate) mod macro_;
pub(crate) mod function;
pub(crate) mod enum_variant;
pub(crate) mod const_;
pub(crate) mod pattern;
pub(crate) mod type_alias;

mod builder_ext;

use base_db::Upcast;
use hir::{
    db::HirDatabase, AsAssocItem, Documentation, HasAttrs, HirDisplay, ModuleDef, Mutability,
    ScopeDef, Type,
};
use ide_db::{
    helpers::{item_name, SnippetCap},
    RootDatabase, SymbolKind,
};
use syntax::TextRange;

use crate::{
    item::{CompletionRelevance, ImportEdit},
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind,
};

use crate::render::{enum_variant::render_variant, function::render_fn, macro_::render_macro};

pub(crate) fn render_field<'a>(
    ctx: RenderContext<'a>,
    field: hir::Field,
    ty: &Type,
) -> CompletionItem {
    Render::new(ctx).add_field(field, ty)
}

pub(crate) fn render_tuple_field<'a>(
    ctx: RenderContext<'a>,
    field: usize,
    ty: &Type,
) -> CompletionItem {
    Render::new(ctx).add_tuple_field(field, ty)
}

pub(crate) fn render_resolution<'a>(
    ctx: RenderContext<'a>,
    local_name: String,
    resolution: &ScopeDef,
) -> Option<CompletionItem> {
    Render::new(ctx).render_resolution(local_name, None, resolution)
}

pub(crate) fn render_resolution_with_import<'a>(
    ctx: RenderContext<'a>,
    import_edit: ImportEdit,
) -> Option<CompletionItem> {
    let resolution = ScopeDef::from(import_edit.import.original_item);
    let local_name = match resolution {
        ScopeDef::ModuleDef(ModuleDef::Function(f)) => f.name(ctx.completion.db).to_string(),
        ScopeDef::ModuleDef(ModuleDef::Const(c)) => c.name(ctx.completion.db)?.to_string(),
        ScopeDef::ModuleDef(ModuleDef::TypeAlias(t)) => t.name(ctx.completion.db).to_string(),
        _ => item_name(ctx.db(), import_edit.import.original_item)?.to_string(),
    };
    Render::new(ctx).render_resolution(local_name, Some(import_edit), &resolution).map(
        |mut item| {
            item.completion_kind = CompletionKind::Magic;
            item
        },
    )
}

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
        self.completion.config.snippet_cap.clone()
    }

    fn db(&self) -> &'a RootDatabase {
        &self.completion.db
    }

    fn source_range(&self) -> TextRange {
        self.completion.source_range()
    }

    fn is_deprecated(&self, node: impl HasAttrs) -> bool {
        let attrs = node.attrs(self.db());
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
            || assoc.containing_trait(db).map(|trait_| self.is_deprecated(trait_)).unwrap_or(false)
    }

    fn docs(&self, node: impl HasAttrs) -> Option<Documentation> {
        node.docs(self.db())
    }

    fn expected_name_and_type(&self) -> Option<(String, Type)> {
        if let Some(record_field) = &self.completion.record_field_syntax {
            cov_mark::hit!(record_field_type_match);
            let (struct_field, _local) = self.completion.sema.resolve_record_field(record_field)?;
            Some((struct_field.name(self.db()).to_string(), struct_field.signature_ty(self.db())))
        } else if let Some(active_parameter) = &self.completion.active_parameter {
            cov_mark::hit!(active_param_type_match);
            Some((active_parameter.name.clone(), active_parameter.ty.clone()))
        } else {
            None
        }
    }
}

/// Generic renderer for completion items.
#[derive(Debug)]
struct Render<'a> {
    ctx: RenderContext<'a>,
}

impl<'a> Render<'a> {
    fn new(ctx: RenderContext<'a>) -> Render<'a> {
        Render { ctx }
    }

    fn add_field(&mut self, field: hir::Field, ty: &Type) -> CompletionItem {
        let is_deprecated = self.ctx.is_deprecated(field);
        let name = field.name(self.ctx.db());
        let mut item = CompletionItem::new(
            CompletionKind::Reference,
            self.ctx.source_range(),
            name.to_string(),
        );
        item.kind(SymbolKind::Field)
            .detail(ty.display(self.ctx.db()).to_string())
            .set_documentation(field.docs(self.ctx.db()))
            .set_deprecated(is_deprecated);

        item.set_relevance(compute_relevance(&self.ctx, &ty, &name.to_string()));

        item.build()
    }

    fn add_tuple_field(&mut self, field: usize, ty: &Type) -> CompletionItem {
        let mut item = CompletionItem::new(
            CompletionKind::Reference,
            self.ctx.source_range(),
            field.to_string(),
        );

        item.kind(SymbolKind::Field).detail(ty.display(self.ctx.db()).to_string());

        item.build()
    }

    fn render_resolution(
        self,
        local_name: String,
        import_to_add: Option<ImportEdit>,
        resolution: &ScopeDef,
    ) -> Option<CompletionItem> {
        let _p = profile::span("render_resolution");
        use hir::ModuleDef::*;

        let completion_kind = match resolution {
            ScopeDef::ModuleDef(BuiltinType(..)) => CompletionKind::BuiltinType,
            _ => CompletionKind::Reference,
        };

        let kind = match resolution {
            ScopeDef::ModuleDef(Function(func)) => {
                return render_fn(self.ctx, import_to_add, Some(local_name), *func);
            }
            ScopeDef::ModuleDef(Variant(_))
                if self.ctx.completion.is_pat_binding_or_const
                    | self.ctx.completion.is_irrefutable_pat_binding =>
            {
                CompletionItemKind::SymbolKind(SymbolKind::Variant)
            }
            ScopeDef::ModuleDef(Variant(var)) => {
                let item = render_variant(self.ctx, import_to_add, Some(local_name), *var, None);
                return Some(item);
            }
            ScopeDef::MacroDef(mac) => {
                let item = render_macro(self.ctx, import_to_add, local_name, *mac);
                return item;
            }

            ScopeDef::ModuleDef(Module(..)) => CompletionItemKind::SymbolKind(SymbolKind::Module),
            ScopeDef::ModuleDef(Adt(adt)) => CompletionItemKind::SymbolKind(match adt {
                hir::Adt::Struct(_) => SymbolKind::Struct,
                hir::Adt::Union(_) => SymbolKind::Union,
                hir::Adt::Enum(_) => SymbolKind::Enum,
            }),
            ScopeDef::ModuleDef(Const(..)) => CompletionItemKind::SymbolKind(SymbolKind::Const),
            ScopeDef::ModuleDef(Static(..)) => CompletionItemKind::SymbolKind(SymbolKind::Static),
            ScopeDef::ModuleDef(Trait(..)) => CompletionItemKind::SymbolKind(SymbolKind::Trait),
            ScopeDef::ModuleDef(TypeAlias(..)) => {
                CompletionItemKind::SymbolKind(SymbolKind::TypeAlias)
            }
            ScopeDef::ModuleDef(BuiltinType(..)) => CompletionItemKind::BuiltinType,
            ScopeDef::GenericParam(param) => CompletionItemKind::SymbolKind(match param {
                hir::GenericParam::TypeParam(_) => SymbolKind::TypeParam,
                hir::GenericParam::LifetimeParam(_) => SymbolKind::LifetimeParam,
                hir::GenericParam::ConstParam(_) => SymbolKind::ConstParam,
            }),
            ScopeDef::Local(..) => CompletionItemKind::SymbolKind(SymbolKind::Local),
            ScopeDef::AdtSelfType(..) | ScopeDef::ImplSelfType(..) => {
                CompletionItemKind::SymbolKind(SymbolKind::SelfParam)
            }
            ScopeDef::Unknown => {
                let mut item = CompletionItem::new(
                    CompletionKind::Reference,
                    self.ctx.source_range(),
                    local_name,
                );
                item.kind(CompletionItemKind::UnresolvedReference).add_import(import_to_add);
                return Some(item.build());
            }
        };

        let mut item =
            CompletionItem::new(completion_kind, self.ctx.source_range(), local_name.clone());
        if let ScopeDef::Local(local) = resolution {
            let ty = local.ty(self.ctx.db());
            if !ty.is_unknown() {
                item.detail(ty.display(self.ctx.db()).to_string());
            }
        };

        if let ScopeDef::Local(local) = resolution {
            let ty = local.ty(self.ctx.db());

            let mut relevance = compute_relevance(&self.ctx, &ty, &local_name);
            relevance.is_local = true;
            item.set_relevance(relevance);

            if let Some((_expected_name, expected_type)) = self.ctx.expected_name_and_type() {
                if ty != expected_type {
                    if let Some(ty_without_ref) = expected_type.remove_ref() {
                        if relevance_type_match(self.ctx.db().upcast(), &ty, &ty_without_ref) {
                            cov_mark::hit!(suggest_ref);
                            let mutability = if expected_type.is_mutable_reference() {
                                Mutability::Mut
                            } else {
                                Mutability::Shared
                            };
                            item.ref_match(mutability);
                        }
                    }
                }
            }
        }

        // Add `<>` for generic types
        if self.ctx.completion.is_path_type
            && !self.ctx.completion.has_type_args
            && self.ctx.completion.config.add_call_parenthesis
        {
            if let Some(cap) = self.ctx.snippet_cap() {
                let has_non_default_type_params = match resolution {
                    ScopeDef::ModuleDef(Adt(it)) => it.has_non_default_type_params(self.ctx.db()),
                    ScopeDef::ModuleDef(TypeAlias(it)) => {
                        it.has_non_default_type_params(self.ctx.db())
                    }
                    _ => false,
                };
                if has_non_default_type_params {
                    cov_mark::hit!(inserts_angle_brackets_for_generics);
                    item.lookup_by(local_name.clone())
                        .label(format!("{}<…>", local_name))
                        .insert_snippet(cap, format!("{}<$0>", local_name));
                }
            }
        }
        item.kind(kind)
            .add_import(import_to_add)
            .set_documentation(self.docs(resolution))
            .set_deprecated(self.is_deprecated(resolution));
        Some(item.build())
    }

    fn docs(&self, resolution: &ScopeDef) -> Option<Documentation> {
        use hir::ModuleDef::*;
        match resolution {
            ScopeDef::ModuleDef(Module(it)) => it.docs(self.ctx.db()),
            ScopeDef::ModuleDef(Adt(it)) => it.docs(self.ctx.db()),
            ScopeDef::ModuleDef(Variant(it)) => it.docs(self.ctx.db()),
            ScopeDef::ModuleDef(Const(it)) => it.docs(self.ctx.db()),
            ScopeDef::ModuleDef(Static(it)) => it.docs(self.ctx.db()),
            ScopeDef::ModuleDef(Trait(it)) => it.docs(self.ctx.db()),
            ScopeDef::ModuleDef(TypeAlias(it)) => it.docs(self.ctx.db()),
            _ => None,
        }
    }

    fn is_deprecated(&self, resolution: &ScopeDef) -> bool {
        match resolution {
            ScopeDef::ModuleDef(it) => self.ctx.is_deprecated_assoc_item(*it),
            ScopeDef::MacroDef(it) => self.ctx.is_deprecated(*it),
            ScopeDef::GenericParam(it) => self.ctx.is_deprecated(*it),
            ScopeDef::AdtSelfType(it) => self.ctx.is_deprecated(*it),
            _ => false,
        }
    }
}

fn compute_relevance(ctx: &RenderContext, ty: &Type, name: &str) -> CompletionRelevance {
    let mut res = CompletionRelevance::default();

    if let Some((expected_name, expected_type)) = ctx.expected_name_and_type() {
        res.exact_type_match = ty == &expected_type;
        res.exact_name_match = name == &expected_name;
    }

    res
}

fn relevance_type_match(db: &dyn HirDatabase, ty: &Type, expected_type: &Type) -> bool {
    ty == expected_type || ty.autoderef(db).any(|deref_ty| &deref_ty == expected_type)
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use itertools::Itertools;

    use crate::{
        test_utils::{check_edit, do_completion, get_all_items, TEST_CONFIG},
        CompletionKind, CompletionRelevance,
    };

    fn check(ra_fixture: &str, expect: Expect) {
        let actual = do_completion(ra_fixture, CompletionKind::Reference);
        expect.assert_debug_eq(&actual);
    }

    fn check_relevance(ra_fixture: &str, expect: Expect) {
        fn display_relevance(relevance: CompletionRelevance) -> String {
            let relevance_factors = vec![
                (relevance.exact_type_match, "type"),
                (relevance.exact_name_match, "name"),
                (relevance.is_local, "local"),
            ]
            .into_iter()
            .filter_map(|(cond, desc)| if cond { Some(desc) } else { None })
            .join("+");

            format!("[{}]", relevance_factors)
        }

        let actual = get_all_items(TEST_CONFIG, ra_fixture)
            .into_iter()
            .filter(|it| it.completion_kind == CompletionKind::Reference)
            .flat_map(|it| {
                let mut items = vec![];

                let tag = it.kind().unwrap().tag();
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
    }

    #[test]
    fn enum_detail_includes_record_fields() {
        check(
            r#"
enum Foo { Foo { x: i32, y: i32 } }

fn main() { Foo::Fo$0 }
"#,
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

fn main() { Foo::Fo$0 }
"#,
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
    fn enum_detail_just_parentheses_for_unit() {
        check(
            r#"
enum Foo { Foo }

fn main() { Foo::Fo$0 }
"#,
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
        check(
            r#"
mod m {
    pub enum Spam { Foo, Bar(i32) }
}
fn main() { let _: m::Spam = S$0 }
"#,
            expect![[r#"
                [
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
                        trigger_call_info: true,
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
                        label: "m::Spam::Foo",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Foo",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Foo",
                        detail: "()",
                    },
                    CompletionItem {
                        label: "main()",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "main()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "-> ()",
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
                        detail: "-> ()",
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
                        detail: "-> ()",
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
                        detail: "-> ()",
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
    fn bar(self) { self.$0 }
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
                        detail: "-> ()",
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

        check(
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
            expect![[r#"
                [
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
            expect![[r#"
                [
                    CompletionItem {
                        label: "the_method()",
                        source_range: 81..81,
                        delete: 81..81,
                        insert: "the_method()$0",
                        kind: Method,
                        lookup: "the_method",
                        detail: "-> ()",
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
        cov_mark::check!(active_param_type_match);
        check_relevance(
            r#"
struct S { foo: i64, bar: u32, baz: u32 }
fn test(bar: u32) { }
fn foo(s: S) { test(s.$0) }
"#,
            expect![[r#"
                fd foo []
                fd bar [type+name]
                fd baz [type]
            "#]],
        );
    }

    #[test]
    fn record_field_relevances() {
        cov_mark::check!(record_field_type_match);
        check_relevance(
            r#"
struct A { foo: i64, bar: u32, baz: u32 }
struct B { x: (), y: f32, bar: u32 }
fn foo(a: A) { B { bar: a.$0 }; }
"#,
            expect![[r#"
                fd foo []
                fd bar [type+name]
                fd baz [type]
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
                fd foo []
                fd bar [type+name]
                fd baz [type]
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
    fn suggest_ref_mut() {
        cov_mark::check!(suggest_ref);
        check(
            r#"
struct S;
fn foo(s: &mut S) {}
fn main() {
    let mut s = S;
    foo($0);
}
            "#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "S",
                        source_range: 70..70,
                        delete: 70..70,
                        insert: "S",
                        kind: SymbolKind(
                            Struct,
                        ),
                    },
                    CompletionItem {
                        label: "foo(…)",
                        source_range: 70..70,
                        delete: 70..70,
                        insert: "foo(${1:&mut s})$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "foo",
                        detail: "-> ()",
                        trigger_call_info: true,
                    },
                    CompletionItem {
                        label: "main()",
                        source_range: 70..70,
                        delete: 70..70,
                        insert: "main()$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "-> ()",
                    },
                    CompletionItem {
                        label: "s",
                        source_range: 70..70,
                        delete: 70..70,
                        insert: "s",
                        kind: SymbolKind(
                            Local,
                        ),
                        detail: "S",
                        relevance: CompletionRelevance {
                            exact_name_match: true,
                            exact_type_match: false,
                            is_local: true,
                        },
                        ref_match: "&mut ",
                    },
                ]
            "#]],
        )
    }

    #[test]
    fn suggest_deref() {
        check_relevance(
            r#"
#[lang = "deref"]
trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct S;
struct T(S);

impl Deref for T {
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
                tt Deref []
                fn foo(…) []
            "#]],
        )
    }

    #[test]
    fn suggest_deref_mut() {
        check_relevance(
            r#"
#[lang = "deref"]
trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

#[lang = "deref_mut"]
pub trait DerefMut: Deref {
    fn deref_mut(&mut self) -> &mut Self::Target;
}

struct S;
struct T(S);

impl Deref for T {
    type Target = S;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for T {
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
                tt DerefMut []
                tt Deref []
                fn foo(…) []
                st T []
                st S []
                fn main() []
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
}
