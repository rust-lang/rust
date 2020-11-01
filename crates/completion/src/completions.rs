//! This module defines an accumulator for completions which are going to be presented to user.

pub(crate) mod attribute;
pub(crate) mod dot;
pub(crate) mod record;
pub(crate) mod pattern;
pub(crate) mod fn_param;
pub(crate) mod keyword;
pub(crate) mod snippet;
pub(crate) mod qualified_path;
pub(crate) mod unqualified_path;
pub(crate) mod postfix;
pub(crate) mod macro_in_item_position;
pub(crate) mod trait_impl;
pub(crate) mod mod_;

use hir::{HasAttrs, HasSource, HirDisplay, ModPath, Mutability, ScopeDef, Type};
use syntax::{ast::NameOwner, display::*};
use test_utils::mark;

use crate::{
    item::Builder,
    render::{ConstRender, EnumVariantRender, FunctionRender, MacroRender},
    CompletionContext, CompletionItem, CompletionItemKind, CompletionKind, CompletionScore,
    RootDatabase,
};

/// Represents an in-progress set of completions being built.
#[derive(Debug, Default)]
pub struct Completions {
    buf: Vec<CompletionItem>,
}

impl Into<Vec<CompletionItem>> for Completions {
    fn into(self) -> Vec<CompletionItem> {
        self.buf
    }
}

impl Builder {
    /// Convenience method, which allows to add a freshly created completion into accumulator
    /// without binding it to the variable.
    pub(crate) fn add_to(self, acc: &mut Completions) {
        acc.add(self.build())
    }
}

impl Completions {
    pub(crate) fn add(&mut self, item: CompletionItem) {
        self.buf.push(item.into())
    }

    pub(crate) fn add_all<I>(&mut self, items: I)
    where
        I: IntoIterator,
        I::Item: Into<CompletionItem>,
    {
        items.into_iter().for_each(|item| self.add(item.into()))
    }

    pub(crate) fn add_field(&mut self, ctx: &CompletionContext, field: hir::Field, ty: &Type) {
        let is_deprecated = is_deprecated(field, ctx.db);
        let name = field.name(ctx.db);
        let mut item =
            CompletionItem::new(CompletionKind::Reference, ctx.source_range(), name.to_string())
                .kind(CompletionItemKind::Field)
                .detail(ty.display(ctx.db).to_string())
                .set_documentation(field.docs(ctx.db))
                .set_deprecated(is_deprecated);

        if let Some(score) = compute_score(ctx, &ty, &name.to_string()) {
            item = item.set_score(score);
        }

        item.add_to(self);
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
                self.add_function(ctx, *func, Some(local_name));
                return;
            }
            ScopeDef::ModuleDef(Adt(hir::Adt::Struct(_))) => CompletionItemKind::Struct,
            // FIXME: add CompletionItemKind::Union
            ScopeDef::ModuleDef(Adt(hir::Adt::Union(_))) => CompletionItemKind::Struct,
            ScopeDef::ModuleDef(Adt(hir::Adt::Enum(_))) => CompletionItemKind::Enum,

            ScopeDef::ModuleDef(EnumVariant(var)) => {
                self.add_enum_variant(ctx, *var, Some(local_name));
                return;
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
                self.add_macro(ctx, Some(local_name), *mac);
                return;
            }
            ScopeDef::Unknown => {
                CompletionItem::new(CompletionKind::Reference, ctx.source_range(), local_name)
                    .kind(CompletionItemKind::UnresolvedReference)
                    .add_to(self);
                return;
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

        let mut item = CompletionItem::new(completion_kind, ctx.source_range(), local_name.clone());
        if let ScopeDef::Local(local) = resolution {
            let ty = local.ty(ctx.db);
            if !ty.is_unknown() {
                item = item.detail(ty.display(ctx.db).to_string());
            }
        };

        let mut ref_match = None;
        if let ScopeDef::Local(local) = resolution {
            if let Some((active_name, active_type)) = ctx.active_name_and_type() {
                let ty = local.ty(ctx.db);
                if let Some(score) =
                    compute_score_from_active(&active_type, &active_name, &ty, &local_name)
                {
                    item = item.set_score(score);
                }
                ref_match = refed_type_matches(&active_type, &active_name, &ty, &local_name);
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
                    item = item
                        .lookup_by(local_name.clone())
                        .label(format!("{}<…>", local_name))
                        .insert_snippet(cap, format!("{}<$0>", local_name));
                }
            }
        }

        item.kind(kind).set_documentation(docs).set_ref_match(ref_match).add_to(self)
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
        if let Some(item) = MacroRender::new(ctx.into(), name, macro_).render() {
            self.add(item);
        }
    }

    pub(crate) fn add_function(
        &mut self,
        ctx: &CompletionContext,
        func: hir::Function,
        local_name: Option<String>,
    ) {
        let item = FunctionRender::new(ctx.into(), local_name, func).render();
        self.add(item)
    }

    pub(crate) fn add_const(&mut self, ctx: &CompletionContext, constant: hir::Const) {
        if let Some(item) = ConstRender::new(ctx.into(), constant).render() {
            self.add(item);
        }
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
        let item = EnumVariantRender::new(ctx.into(), None, variant, Some(path)).render();
        self.add(item);
    }

    pub(crate) fn add_enum_variant(
        &mut self,
        ctx: &CompletionContext,
        variant: hir::EnumVariant,
        local_name: Option<String>,
    ) {
        let item = EnumVariantRender::new(ctx.into(), local_name, variant, None).render();
        self.add(item);
    }
}

fn compute_score_from_active(
    active_type: &Type,
    active_name: &str,
    ty: &Type,
    name: &str,
) -> Option<CompletionScore> {
    // Compute score
    // For the same type
    if active_type != ty {
        return None;
    }

    let mut res = CompletionScore::TypeMatch;

    // If same type + same name then go top position
    if active_name == name {
        res = CompletionScore::TypeAndNameMatch
    }

    Some(res)
}
fn refed_type_matches(
    active_type: &Type,
    active_name: &str,
    ty: &Type,
    name: &str,
) -> Option<(Mutability, CompletionScore)> {
    let derefed_active = active_type.remove_ref()?;
    let score = compute_score_from_active(&derefed_active, &active_name, &ty, &name)?;
    Some((
        if active_type.is_mutable_reference() { Mutability::Mut } else { Mutability::Shared },
        score,
    ))
}

fn compute_score(ctx: &CompletionContext, ty: &Type, name: &str) -> Option<CompletionScore> {
    let (active_name, active_type) = ctx.active_name_and_type()?;
    compute_score_from_active(&active_type, &active_name, ty, name)
}

fn is_deprecated(node: impl HasAttrs, db: &RootDatabase) -> bool {
    node.attrs(db).by_key("deprecated").exists()
}

#[cfg(test)]
mod tests {
    use std::cmp::Reverse;

    use expect_test::{expect, Expect};
    use test_utils::mark;

    use crate::{
        test_utils::{check_edit, check_edit_with_config, do_completion, get_all_items},
        CompletionConfig, CompletionKind, CompletionScore,
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

        let mut completions = get_all_items(CompletionConfig::default(), ra_fixture);
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
    fn lookup_enums_by_two_qualifiers() {
        check(
            r#"
mod m {
    pub enum Spam { Foo, Bar(i32) }
}
fn main() { let _: m::Spam = S<|> }
"#,
            expect![[r#"
                [
                    CompletionItem {
                        label: "Spam::Bar(…)",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "Spam::Bar($0)",
                        kind: EnumVariant,
                        lookup: "Spam::Bar",
                        detail: "(i32)",
                        trigger_call_info: true,
                    },
                    CompletionItem {
                        label: "m",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m",
                        kind: Module,
                    },
                    CompletionItem {
                        label: "m::Spam::Foo",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Foo",
                        kind: EnumVariant,
                        lookup: "Spam::Foo",
                        detail: "()",
                    },
                    CompletionItem {
                        label: "main()",
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "main()$0",
                        kind: Function,
                        lookup: "main",
                        detail: "fn main()",
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
            CompletionConfig { add_call_argument_snippets: false, ..CompletionConfig::default() },
            "with_args",
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_<|> }
"#,
            r#"
fn with_args(x: i32, y: String) {}
fn main() { with_args($0) }
"#,
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
    fn insert_ref_when_matching_local_in_scope() {
        check_edit(
            "ref_arg",
            r#"
struct Foo {}
fn ref_arg(x: &Foo) {}
fn main() {
    let x = Foo {};
    ref_ar<|>
}
"#,
            r#"
struct Foo {}
fn ref_arg(x: &Foo) {}
fn main() {
    let x = Foo {};
    ref_arg(${1:&x})$0
}
"#,
        );
    }

    #[test]
    fn insert_mut_ref_when_matching_local_in_scope() {
        check_edit(
            "ref_arg",
            r#"
struct Foo {}
fn ref_arg(x: &mut Foo) {}
fn main() {
    let x = Foo {};
    ref_ar<|>
}
"#,
            r#"
struct Foo {}
fn ref_arg(x: &mut Foo) {}
fn main() {
    let x = Foo {};
    ref_arg(${1:&mut x})$0
}
"#,
        );
    }

    #[test]
    fn insert_ref_when_matching_local_in_scope_for_method() {
        check_edit(
            "apply_foo",
            r#"
struct Foo {}
struct Bar {}
impl Bar {
    fn apply_foo(&self, x: &Foo) {}
}

fn main() {
    let x = Foo {};
    let y = Bar {};
    y.<|>
}
"#,
            r#"
struct Foo {}
struct Bar {}
impl Bar {
    fn apply_foo(&self, x: &Foo) {}
}

fn main() {
    let x = Foo {};
    let y = Bar {};
    y.apply_foo(${1:&x})$0
}
"#,
        );
    }

    #[test]
    fn trim_mut_keyword_in_func_completion() {
        check_edit(
            "take_mutably",
            r#"
fn take_mutably(mut x: &i32) {}

fn main() {
    take_m<|>
}
"#,
            r#"
fn take_mutably(mut x: &i32) {}

fn main() {
    take_mutably(${1:x})$0
}
"#,
        );
    }

    #[test]
    fn inserts_parens_for_tuple_enums() {
        mark::check!(inserts_parens_for_tuple_enums);
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
    fn dont_duplicate_pattern_parens() {
        mark::check!(dont_duplicate_pattern_parens);
        check_edit(
            "Var",
            r#"
enum E { Var(i32) }
fn main() {
    match E::Var(92) {
        E::<|>(92) => (),
    }
}
"#,
            r#"
enum E { Var(i32) }
fn main() {
    match E::Var(92) {
        E::Var(92) => (),
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
//- /main.rs crate:main deps:foo
use foo::<|>;
//- /foo/lib.rs crate:foo
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
    fn too_many_arguments() {
        check_scores(
            r#"
struct Foo;
fn f(foo: &Foo) { f(foo, w<|>) }
"#,
            expect![[r#"
                st Foo []
                fn f(…) []
                bn foo []
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
