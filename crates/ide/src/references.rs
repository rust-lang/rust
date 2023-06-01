//! This module implements a reference search.
//! First, the element at the cursor position must be either an `ast::Name`
//! or `ast::NameRef`. If it's an `ast::NameRef`, at the classification step we
//! try to resolve the direct tree parent of this element, otherwise we
//! already have a definition and just need to get its HIR together with
//! some information that is needed for further steps of searching.
//! After that, we collect files that might contain references and look
//! for text occurrences of the identifier. If there's an `ast::NameRef`
//! at the index that the match starts at and its tree parent is
//! resolved to the search element definition, we get a reference.

use hir::{PathResolution, Semantics};
use ide_db::{
    base_db::FileId,
    defs::{Definition, NameClass, NameRefClass},
    search::{ReferenceCategory, SearchScope, UsageSearchResult},
    RootDatabase,
};
use itertools::Itertools;
use nohash_hasher::IntMap;
use syntax::{
    algo::find_node_at_offset,
    ast::{self, HasName},
    match_ast, AstNode,
    SyntaxKind::*,
    SyntaxNode, TextRange, TextSize, T,
};

use crate::{FilePosition, NavigationTarget, TryToNav};

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    pub declaration: Option<Declaration>,
    pub references: IntMap<FileId, Vec<(TextRange, Option<ReferenceCategory>)>>,
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub nav: NavigationTarget,
    pub is_mut: bool,
}

// Feature: Find All References
//
// Shows all references of the item at the cursor location
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[Shift+Alt+F12]
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113020670-b7c34f00-917a-11eb-8003-370ac5f2b3cb.gif[]
pub(crate) fn find_all_refs(
    sema: &Semantics<'_, RootDatabase>,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Option<Vec<ReferenceSearchResult>> {
    let _p = profile::span("find_all_refs");
    let syntax = sema.parse(position.file_id).syntax().clone();
    let make_searcher = |literal_search: bool| {
        move |def: Definition| {
            let declaration = match def {
                Definition::Module(module) => {
                    Some(NavigationTarget::from_module_to_decl(sema.db, module))
                }
                def => def.try_to_nav(sema.db),
            }
            .map(|nav| {
                let decl_range = nav.focus_or_full_range();
                Declaration {
                    is_mut: decl_mutability(&def, sema.parse(nav.file_id).syntax(), decl_range),
                    nav,
                }
            });
            let mut usages =
                def.usages(sema).set_scope(search_scope.clone()).include_self_refs().all();

            if literal_search {
                retain_adt_literal_usages(&mut usages, def, sema);
            }

            let references = usages
                .into_iter()
                .map(|(file_id, refs)| {
                    (
                        file_id,
                        refs.into_iter()
                            .map(|file_ref| (file_ref.range, file_ref.category))
                            .unique()
                            .collect(),
                    )
                })
                .collect();

            ReferenceSearchResult { declaration, references }
        }
    };

    match name_for_constructor_search(&syntax, position) {
        Some(name) => {
            let def = match NameClass::classify(sema, &name)? {
                NameClass::Definition(it) | NameClass::ConstReference(it) => it,
                NameClass::PatFieldShorthand { local_def: _, field_ref } => {
                    Definition::Field(field_ref)
                }
            };
            Some(vec![make_searcher(true)(def)])
        }
        None => {
            let search = make_searcher(false);
            Some(find_defs(sema, &syntax, position.offset)?.map(search).collect())
        }
    }
}

pub(crate) fn find_defs<'a>(
    sema: &'a Semantics<'_, RootDatabase>,
    syntax: &SyntaxNode,
    offset: TextSize,
) -> Option<impl Iterator<Item = Definition> + 'a> {
    let token = syntax.token_at_offset(offset).find(|t| {
        matches!(
            t.kind(),
            IDENT | INT_NUMBER | LIFETIME_IDENT | T![self] | T![super] | T![crate] | T![Self]
        )
    });
    token.map(|token| {
        sema.descend_into_macros_with_same_text(token)
            .into_iter()
            .filter_map(|it| ast::NameLike::cast(it.parent()?))
            .filter_map(move |name_like| {
                let def = match name_like {
                    ast::NameLike::NameRef(name_ref) => {
                        match NameRefClass::classify(sema, &name_ref)? {
                            NameRefClass::Definition(def) => def,
                            NameRefClass::FieldShorthand { local_ref, field_ref: _ } => {
                                Definition::Local(local_ref)
                            }
                        }
                    }
                    ast::NameLike::Name(name) => match NameClass::classify(sema, &name)? {
                        NameClass::Definition(it) | NameClass::ConstReference(it) => it,
                        NameClass::PatFieldShorthand { local_def, field_ref: _ } => {
                            Definition::Local(local_def)
                        }
                    },
                    ast::NameLike::Lifetime(lifetime) => {
                        NameRefClass::classify_lifetime(sema, &lifetime)
                            .and_then(|class| match class {
                                NameRefClass::Definition(it) => Some(it),
                                _ => None,
                            })
                            .or_else(|| {
                                NameClass::classify_lifetime(sema, &lifetime)
                                    .and_then(NameClass::defined)
                            })?
                    }
                };
                Some(def)
            })
    })
}

pub(crate) fn decl_mutability(def: &Definition, syntax: &SyntaxNode, range: TextRange) -> bool {
    match def {
        Definition::Local(_) | Definition::Field(_) => {}
        _ => return false,
    };

    match find_node_at_offset::<ast::LetStmt>(syntax, range.start()) {
        Some(stmt) if stmt.initializer().is_some() => match stmt.pat() {
            Some(ast::Pat::IdentPat(it)) => it.mut_token().is_some(),
            _ => false,
        },
        _ => false,
    }
}

/// Filter out all non-literal usages for adt-defs
fn retain_adt_literal_usages(
    usages: &mut UsageSearchResult,
    def: Definition,
    sema: &Semantics<'_, RootDatabase>,
) {
    let refs = usages.references.values_mut();
    match def {
        Definition::Adt(hir::Adt::Enum(enum_)) => {
            refs.for_each(|it| {
                it.retain(|reference| {
                    reference
                        .name
                        .as_name_ref()
                        .map_or(false, |name_ref| is_enum_lit_name_ref(sema, enum_, name_ref))
                })
            });
            usages.references.retain(|_, it| !it.is_empty());
        }
        Definition::Adt(_) | Definition::Variant(_) => {
            refs.for_each(|it| {
                it.retain(|reference| reference.name.as_name_ref().map_or(false, is_lit_name_ref))
            });
            usages.references.retain(|_, it| !it.is_empty());
        }
        _ => {}
    }
}

/// Returns `Some` if the cursor is at a position for an item to search for all its constructor/literal usages
fn name_for_constructor_search(syntax: &SyntaxNode, position: FilePosition) -> Option<ast::Name> {
    let token = syntax.token_at_offset(position.offset).right_biased()?;
    let token_parent = token.parent()?;
    let kind = token.kind();
    if kind == T![;] {
        ast::Struct::cast(token_parent)
            .filter(|struct_| struct_.field_list().is_none())
            .and_then(|struct_| struct_.name())
    } else if kind == T!['{'] {
        match_ast! {
            match token_parent {
                ast::RecordFieldList(rfl) => match_ast! {
                    match (rfl.syntax().parent()?) {
                        ast::Variant(it) => it.name(),
                        ast::Struct(it) => it.name(),
                        ast::Union(it) => it.name(),
                        _ => None,
                    }
                },
                ast::VariantList(vl) => ast::Enum::cast(vl.syntax().parent()?)?.name(),
                _ => None,
            }
        }
    } else if kind == T!['('] {
        let tfl = ast::TupleFieldList::cast(token_parent)?;
        match_ast! {
            match (tfl.syntax().parent()?) {
                ast::Variant(it) => it.name(),
                ast::Struct(it) => it.name(),
                _ => None,
            }
        }
    } else {
        None
    }
}

fn is_enum_lit_name_ref(
    sema: &Semantics<'_, RootDatabase>,
    enum_: hir::Enum,
    name_ref: &ast::NameRef,
) -> bool {
    let path_is_variant_of_enum = |path: ast::Path| {
        matches!(
            sema.resolve_path(&path),
            Some(PathResolution::Def(hir::ModuleDef::Variant(variant)))
                if variant.parent_enum(sema.db) == enum_
        )
    };
    name_ref
        .syntax()
        .ancestors()
        .find_map(|ancestor| {
            match_ast! {
                match ancestor {
                    ast::PathExpr(path_expr) => path_expr.path().map(path_is_variant_of_enum),
                    ast::RecordExpr(record_expr) => record_expr.path().map(path_is_variant_of_enum),
                    _ => None,
                }
            }
        })
        .unwrap_or(false)
}

fn path_ends_with(path: Option<ast::Path>, name_ref: &ast::NameRef) -> bool {
    path.and_then(|path| path.segment())
        .and_then(|segment| segment.name_ref())
        .map_or(false, |segment| segment == *name_ref)
}

fn is_lit_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref.syntax().ancestors().find_map(|ancestor| {
        match_ast! {
            match ancestor {
                ast::PathExpr(path_expr) => Some(path_ends_with(path_expr.path(), name_ref)),
                ast::RecordExpr(record_expr) => Some(path_ends_with(record_expr.path(), name_ref)),
                _ => None,
            }
        }
    }).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide_db::{base_db::FileId, search::ReferenceCategory};
    use stdx::format_to;

    use crate::{fixture, SearchScope};

    #[test]
    fn test_struct_literal_after_space() {
        check(
            r#"
struct Foo $0{
    a: i32,
}
impl Foo {
    fn f() -> i32 { 42 }
}
fn main() {
    let f: Foo;
    f = Foo {a: Foo::f()};
}
"#,
            expect![[r#"
                Foo Struct FileId(0) 0..26 7..10

                FileId(0) 101..104
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_before_space() {
        check(
            r#"
struct Foo$0 {}
    fn main() {
    let f: Foo;
    f = Foo {};
}
"#,
            expect![[r#"
                Foo Struct FileId(0) 0..13 7..10

                FileId(0) 41..44
                FileId(0) 54..57
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_with_generic_type() {
        check(
            r#"
struct Foo<T> $0{}
    fn main() {
    let f: Foo::<i32>;
    f = Foo {};
}
"#,
            expect![[r#"
                Foo Struct FileId(0) 0..16 7..10

                FileId(0) 64..67
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_for_tuple() {
        check(
            r#"
struct Foo$0(i32);

fn main() {
    let f: Foo;
    f = Foo(1);
}
"#,
            expect![[r#"
                Foo Struct FileId(0) 0..16 7..10

                FileId(0) 54..57
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_for_union() {
        check(
            r#"
union Foo $0{
    x: u32
}

fn main() {
    let f: Foo;
    f = Foo { x: 1 };
}
"#,
            expect![[r#"
                Foo Union FileId(0) 0..24 6..9

                FileId(0) 62..65
            "#]],
        );
    }

    #[test]
    fn test_enum_after_space() {
        check(
            r#"
enum Foo $0{
    A,
    B(),
    C{},
}
fn main() {
    let f: Foo;
    f = Foo::A;
    f = Foo::B();
    f = Foo::C{};
}
"#,
            expect![[r#"
                Foo Enum FileId(0) 0..37 5..8

                FileId(0) 74..77
                FileId(0) 90..93
                FileId(0) 108..111
            "#]],
        );
    }

    #[test]
    fn test_variant_record_after_space() {
        check(
            r#"
enum Foo {
    A $0{ n: i32 },
    B,
}
fn main() {
    let f: Foo;
    f = Foo::B;
    f = Foo::A { n: 92 };
}
"#,
            expect![[r#"
                A Variant FileId(0) 15..27 15..16

                FileId(0) 95..96
            "#]],
        );
    }
    #[test]
    fn test_variant_tuple_before_paren() {
        check(
            r#"
enum Foo {
    A$0(i32),
    B,
}
fn main() {
    let f: Foo;
    f = Foo::B;
    f = Foo::A(92);
}
"#,
            expect![[r#"
                A Variant FileId(0) 15..21 15..16

                FileId(0) 89..90
            "#]],
        );
    }

    #[test]
    fn test_enum_before_space() {
        check(
            r#"
enum Foo$0 {
    A,
    B,
}
fn main() {
    let f: Foo;
    f = Foo::A;
}
"#,
            expect![[r#"
                Foo Enum FileId(0) 0..26 5..8

                FileId(0) 50..53
                FileId(0) 63..66
            "#]],
        );
    }

    #[test]
    fn test_enum_with_generic_type() {
        check(
            r#"
enum Foo<T> $0{
    A(T),
    B,
}
fn main() {
    let f: Foo<i8>;
    f = Foo::A(1);
}
"#,
            expect![[r#"
                Foo Enum FileId(0) 0..32 5..8

                FileId(0) 73..76
            "#]],
        );
    }

    #[test]
    fn test_enum_for_tuple() {
        check(
            r#"
enum Foo$0{
    A(i8),
    B(i8),
}
fn main() {
    let f: Foo;
    f = Foo::A(1);
}
"#,
            expect![[r#"
                Foo Enum FileId(0) 0..33 5..8

                FileId(0) 70..73
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_for_local() {
        check(
            r#"
fn main() {
    let mut i = 1;
    let j = 1;
    i = i$0 + j;

    {
        i = 0;
    }

    i = 5;
}"#,
            expect![[r#"
                i Local FileId(0) 20..25 24..25 Write

                FileId(0) 50..51 Write
                FileId(0) 54..55 Read
                FileId(0) 76..77 Write
                FileId(0) 94..95 Write
            "#]],
        );
    }

    #[test]
    fn search_filters_by_range() {
        check(
            r#"
fn foo() {
    let spam$0 = 92;
    spam + spam
}
fn bar() {
    let spam = 92;
    spam + spam
}
"#,
            expect![[r#"
                spam Local FileId(0) 19..23 19..23

                FileId(0) 34..38 Read
                FileId(0) 41..45 Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_for_param_inside() {
        check(
            r#"
fn foo(i : u32) -> u32 { i$0 }
"#,
            expect![[r#"
                i ValueParam FileId(0) 7..8 7..8

                FileId(0) 25..26 Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_for_fn_param() {
        check(
            r#"
fn foo(i$0 : u32) -> u32 { i }
"#,
            expect![[r#"
                i ValueParam FileId(0) 7..8 7..8

                FileId(0) 25..26 Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_field_name() {
        check(
            r#"
//- /lib.rs
struct Foo {
    pub spam$0: u32,
}

fn main(s: Foo) {
    let f = s.spam;
}
"#,
            expect![[r#"
                spam Field FileId(0) 17..30 21..25

                FileId(0) 67..71 Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_impl_item_name() {
        check(
            r#"
struct Foo;
impl Foo {
    fn f$0(&self) {  }
}
"#,
            expect![[r#"
                f Function FileId(0) 27..43 30..31

                (no references)
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_name() {
        check(
            r#"
enum Foo {
    A,
    B$0,
    C,
}
"#,
            expect![[r#"
                B Variant FileId(0) 22..23 22..23

                (no references)
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_field() {
        check(
            r#"
enum Foo {
    A,
    B { field$0: u8 },
    C,
}
"#,
            expect![[r#"
                field Field FileId(0) 26..35 26..31

                (no references)
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_two_modules() {
        check(
            r#"
//- /lib.rs
pub mod foo;
pub mod bar;

fn f() {
    let i = foo::Foo { n: 5 };
}

//- /foo.rs
use crate::bar;

pub struct Foo {
    pub n: u32,
}

fn f() {
    let i = bar::Bar { n: 5 };
}

//- /bar.rs
use crate::foo;

pub struct Bar {
    pub n: u32,
}

fn f() {
    let i = foo::Foo$0 { n: 5 };
}
"#,
            expect![[r#"
                Foo Struct FileId(1) 17..51 28..31 foo

                FileId(0) 53..56
                FileId(2) 79..82
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_decl_module() {
        check(
            r#"
//- /lib.rs
mod foo$0;

use foo::Foo;

fn f() {
    let i = Foo { n: 5 };
}

//- /foo.rs
pub struct Foo {
    pub n: u32,
}
"#,
            expect![[r#"
                foo Module FileId(0) 0..8 4..7

                FileId(0) 14..17 Import
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_decl_module_on_self() {
        check(
            r#"
//- /lib.rs
mod foo;

//- /foo.rs
use self$0;
"#,
            expect![[r#"
                foo Module FileId(0) 0..8 4..7

                FileId(1) 4..8 Import
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_decl_module_on_self_crate_root() {
        check(
            r#"
//- /lib.rs
use self$0;
"#,
            expect![[r#"
                Module FileId(0) 0..10

                FileId(0) 4..8 Import
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_super_mod_vis() {
        check(
            r#"
//- /lib.rs
mod foo;

//- /foo.rs
mod some;
use some::Foo;

fn f() {
    let i = Foo { n: 5 };
}

//- /foo/some.rs
pub(super) struct Foo$0 {
    pub n: u32,
}
"#,
            expect![[r#"
                Foo Struct FileId(2) 0..41 18..21 some

                FileId(1) 20..23 Import
                FileId(1) 47..50
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_with_scope() {
        let code = r#"
            //- /lib.rs
            mod foo;
            mod bar;

            pub fn quux$0() {}

            //- /foo.rs
            fn f() { super::quux(); }

            //- /bar.rs
            fn f() { super::quux(); }
        "#;

        check_with_scope(
            code,
            None,
            expect![[r#"
                quux Function FileId(0) 19..35 26..30

                FileId(1) 16..20
                FileId(2) 16..20
            "#]],
        );

        check_with_scope(
            code,
            Some(SearchScope::single_file(FileId(2))),
            expect![[r#"
                quux Function FileId(0) 19..35 26..30

                FileId(2) 16..20
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_macro_def() {
        check(
            r#"
#[macro_export]
macro_rules! m1$0 { () => (()) }

fn foo() {
    m1();
    m1();
}
"#,
            expect![[r#"
                m1 Macro FileId(0) 0..46 29..31

                FileId(0) 63..65
                FileId(0) 73..75
            "#]],
        );
    }

    #[test]
    fn test_basic_highlight_read_write() {
        check(
            r#"
fn foo() {
    let mut i$0 = 0;
    i = i + 1;
}
"#,
            expect![[r#"
                i Local FileId(0) 19..24 23..24 Write

                FileId(0) 34..35 Write
                FileId(0) 38..39 Read
            "#]],
        );
    }

    #[test]
    fn test_basic_highlight_field_read_write() {
        check(
            r#"
struct S {
    f: u32,
}

fn foo() {
    let mut s = S{f: 0};
    s.f$0 = 0;
}
"#,
            expect![[r#"
                f Field FileId(0) 15..21 15..16

                FileId(0) 55..56 Read
                FileId(0) 68..69 Write
            "#]],
        );
    }

    #[test]
    fn test_basic_highlight_decl_no_write() {
        check(
            r#"
fn foo() {
    let i$0;
    i = 1;
}
"#,
            expect![[r#"
                i Local FileId(0) 19..20 19..20

                FileId(0) 26..27 Write
            "#]],
        );
    }

    #[test]
    fn test_find_struct_function_refs_outside_module() {
        check(
            r#"
mod foo {
    pub struct Foo;

    impl Foo {
        pub fn new$0() -> Foo { Foo }
    }
}

fn main() {
    let _f = foo::Foo::new();
}
"#,
            expect![[r#"
                new Function FileId(0) 54..81 61..64

                FileId(0) 126..129
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_nested_module() {
        check(
            r#"
//- /lib.rs
mod foo { mod bar; }

fn f$0() {}

//- /foo/bar.rs
use crate::f;

fn g() { f(); }
"#,
            expect![[r#"
                f Function FileId(0) 22..31 25..26

                FileId(1) 11..12 Import
                FileId(1) 24..25
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_struct_pat() {
        check(
            r#"
struct S {
    field$0: u8,
}

fn f(s: S) {
    match s {
        S { field } => {}
    }
}
"#,
            expect![[r#"
                field Field FileId(0) 15..24 15..20

                FileId(0) 68..73 Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_pat() {
        check(
            r#"
enum En {
    Variant {
        field$0: u8,
    }
}

fn f(e: En) {
    match e {
        En::Variant { field } => {}
    }
}
"#,
            expect![[r#"
                field Field FileId(0) 32..41 32..37

                FileId(0) 102..107 Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_privacy() {
        check(
            r#"
mod m {
    pub enum En {
        Variant {
            field$0: u8,
        }
    }
}

fn f() -> m::En {
    m::En::Variant { field: 0 }
}
"#,
            expect![[r#"
                field Field FileId(0) 56..65 56..61

                FileId(0) 125..130 Read
            "#]],
        );
    }

    #[test]
    fn test_find_self_refs() {
        check(
            r#"
struct Foo { bar: i32 }

impl Foo {
    fn foo(self) {
        let x = self$0.bar;
        if true {
            let _ = match () {
                () => self,
            };
        }
    }
}
"#,
            expect![[r#"
                self SelfParam FileId(0) 47..51 47..51

                FileId(0) 71..75 Read
                FileId(0) 152..156 Read
            "#]],
        );
    }

    #[test]
    fn test_find_self_refs_decl() {
        check(
            r#"
struct Foo { bar: i32 }

impl Foo {
    fn foo(self$0) {
        self;
    }
}
"#,
            expect![[r#"
                self SelfParam FileId(0) 47..51 47..51

                FileId(0) 63..67 Read
            "#]],
        );
    }

    fn check(ra_fixture: &str, expect: Expect) {
        check_with_scope(ra_fixture, None, expect)
    }

    fn check_with_scope(ra_fixture: &str, search_scope: Option<SearchScope>, expect: Expect) {
        let (analysis, pos) = fixture::position(ra_fixture);
        let refs = analysis.find_all_refs(pos, search_scope).unwrap().unwrap();

        let mut actual = String::new();
        for refs in refs {
            actual += "\n\n";

            if let Some(decl) = refs.declaration {
                format_to!(actual, "{}", decl.nav.debug_render());
                if decl.is_mut {
                    format_to!(actual, " {:?}", ReferenceCategory::Write)
                }
                actual += "\n\n";
            }

            for (file_id, references) in &refs.references {
                for (range, access) in references {
                    format_to!(actual, "{:?} {:?}", file_id, range);
                    if let Some(access) = access {
                        format_to!(actual, " {:?}", access);
                    }
                    actual += "\n";
                }
            }

            if refs.references.is_empty() {
                actual += "(no references)\n";
            }
        }
        expect.assert_eq(actual.trim_start())
    }

    #[test]
    fn test_find_lifetimes_function() {
        check(
            r#"
trait Foo<'a> {}
impl<'a> Foo<'a> for &'a () {}
fn foo<'a, 'b: 'a>(x: &'a$0 ()) -> &'a () where &'a (): Foo<'a> {
    fn bar<'a>(_: &'a ()) {}
    x
}
"#,
            expect![[r#"
                'a LifetimeParam FileId(0) 55..57 55..57

                FileId(0) 63..65
                FileId(0) 71..73
                FileId(0) 82..84
                FileId(0) 95..97
                FileId(0) 106..108
            "#]],
        );
    }

    #[test]
    fn test_find_lifetimes_type_alias() {
        check(
            r#"
type Foo<'a, T> where T: 'a$0 = &'a T;
"#,
            expect![[r#"
                'a LifetimeParam FileId(0) 9..11 9..11

                FileId(0) 25..27
                FileId(0) 31..33
            "#]],
        );
    }

    #[test]
    fn test_find_lifetimes_trait_impl() {
        check(
            r#"
trait Foo<'a> {
    fn foo() -> &'a ();
}
impl<'a> Foo<'a> for &'a () {
    fn foo() -> &'a$0 () {
        unimplemented!()
    }
}
"#,
            expect![[r#"
                'a LifetimeParam FileId(0) 47..49 47..49

                FileId(0) 55..57
                FileId(0) 64..66
                FileId(0) 89..91
            "#]],
        );
    }

    #[test]
    fn test_map_range_to_original() {
        check(
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let a$0 = "test";
    foo!(a);
}
"#,
            expect![[r#"
                a Local FileId(0) 59..60 59..60

                FileId(0) 80..81 Read
            "#]],
        );
    }

    #[test]
    fn test_map_range_to_original_ref() {
        check(
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let a = "test";
    foo!(a$0);
}
"#,
            expect![[r#"
                a Local FileId(0) 59..60 59..60

                FileId(0) 80..81 Read
            "#]],
        );
    }

    #[test]
    fn test_find_labels() {
        check(
            r#"
fn foo<'a>() -> &'a () {
    'a: loop {
        'b: loop {
            continue 'a$0;
        }
        break 'a;
    }
}
"#,
            expect![[r#"
                'a Label FileId(0) 29..32 29..31

                FileId(0) 80..82
                FileId(0) 108..110
            "#]],
        );
    }

    #[test]
    fn test_find_const_param() {
        check(
            r#"
fn foo<const FOO$0: usize>() -> usize {
    FOO
}
"#,
            expect![[r#"
                FOO ConstParam FileId(0) 7..23 13..16

                FileId(0) 42..45
            "#]],
        );
    }

    #[test]
    fn test_trait() {
        check(
            r#"
trait Foo$0 where Self: {}

impl Foo for () {}
"#,
            expect![[r#"
                Foo Trait FileId(0) 0..24 6..9

                FileId(0) 31..34
            "#]],
        );
    }

    #[test]
    fn test_trait_self() {
        check(
            r#"
trait Foo where Self$0 {
    fn f() -> Self;
}

impl Foo for () {}
"#,
            expect![[r#"
                Self TypeParam FileId(0) 0..44 6..9

                FileId(0) 16..20
                FileId(0) 37..41
            "#]],
        );
    }

    #[test]
    fn test_self_ty() {
        check(
            r#"
        struct $0Foo;

        impl Foo where Self: {
            fn f() -> Self;
        }
        "#,
            expect![[r#"
                Foo Struct FileId(0) 0..11 7..10

                FileId(0) 18..21
                FileId(0) 28..32
                FileId(0) 50..54
            "#]],
        );
        check(
            r#"
struct Foo;

impl Foo where Self: {
    fn f() -> Self$0;
}
"#,
            expect![[r#"
                impl Impl FileId(0) 13..57 18..21

                FileId(0) 18..21
                FileId(0) 28..32
                FileId(0) 50..54
            "#]],
        );
    }
    #[test]
    fn test_self_variant_with_payload() {
        check(
            r#"
enum Foo { Bar() }

impl Foo {
    fn foo(self) {
        match self {
            Self::Bar$0() => (),
        }
    }
}

"#,
            expect![[r#"
                Bar Variant FileId(0) 11..16 11..14

                FileId(0) 89..92
            "#]],
        );
    }

    #[test]
    fn test_trait_alias() {
        check(
            r#"
trait Foo {}
trait Bar$0 = Foo where Self: ;
fn foo<T: Bar>(_: impl Bar, _: &dyn Bar) {}
"#,
            expect![[r#"
                Bar TraitAlias FileId(0) 13..42 19..22

                FileId(0) 53..56
                FileId(0) 66..69
                FileId(0) 79..82
            "#]],
        );
    }

    #[test]
    fn test_trait_alias_self() {
        check(
            r#"
trait Foo = where Self$0: ;
"#,
            expect![[r#"
                Self TypeParam FileId(0) 0..25 6..9

                FileId(0) 18..22
            "#]],
        );
    }

    #[test]
    fn test_attr_differs_from_fn_with_same_name() {
        check(
            r#"
#[test]
fn test$0() {
    test();
}
"#,
            expect![[r#"
                test Function FileId(0) 0..33 11..15

                FileId(0) 24..28
            "#]],
        );
    }

    #[test]
    fn test_const_in_pattern() {
        check(
            r#"
const A$0: i32 = 42;

fn main() {
    match A {
        A => (),
        _ => (),
    }
    if let A = A {}
}
"#,
            expect![[r#"
                A Const FileId(0) 0..18 6..7

                FileId(0) 42..43
                FileId(0) 54..55
                FileId(0) 97..98
                FileId(0) 101..102
            "#]],
        );
    }

    #[test]
    fn test_primitives() {
        check(
            r#"
fn foo(_: bool) -> bo$0ol { true }
"#,
            expect![[r#"
                FileId(0) 10..14
                FileId(0) 19..23
            "#]],
        );
    }

    #[test]
    fn test_transitive() {
        check(
            r#"
//- /level3.rs new_source_root:local crate:level3
pub struct Fo$0o;
//- /level2.rs new_source_root:local crate:level2 deps:level3
pub use level3::Foo;
//- /level1.rs new_source_root:local crate:level1 deps:level2
pub use level2::Foo;
//- /level0.rs new_source_root:local crate:level0 deps:level1
pub use level1::Foo;
"#,
            expect![[r#"
                Foo Struct FileId(0) 0..15 11..14

                FileId(1) 16..19 Import
                FileId(2) 16..19 Import
                FileId(3) 16..19 Import
            "#]],
        );
    }

    #[test]
    fn test_decl_macro_references() {
        check(
            r#"
//- /lib.rs crate:lib
#[macro_use]
mod qux;
mod bar;

pub use self::foo;
//- /qux.rs
#[macro_export]
macro_rules! foo$0 {
    () => {struct Foo;};
}
//- /bar.rs
foo!();
//- /other.rs crate:other deps:lib new_source_root:local
lib::foo!();
"#,
            expect![[r#"
                foo Macro FileId(1) 0..61 29..32

                FileId(0) 46..49 Import
                FileId(2) 0..3
                FileId(3) 5..8
            "#]],
        );
    }

    #[test]
    fn macro_doesnt_reference_attribute_on_call() {
        check(
            r#"
macro_rules! m {
    () => {};
}

#[proc_macro_test::attr_noop]
m$0!();

"#,
            expect![[r#"
                m Macro FileId(0) 0..32 13..14

                FileId(0) 64..65
            "#]],
        );
    }

    #[test]
    fn multi_def() {
        check(
            r#"
macro_rules! m {
    ($name:ident) => {
        mod module {
            pub fn $name() {}
        }

        pub fn $name() {}
    }
}

m!(func$0);

fn f() {
    func();
    module::func();
}
            "#,
            expect![[r#"
                func Function FileId(0) 137..146 140..144

                FileId(0) 161..165


                func Function FileId(0) 137..146 140..144 module

                FileId(0) 181..185
            "#]],
        )
    }

    #[test]
    fn attr_expanded() {
        check(
            r#"
//- proc_macros: identity
#[proc_macros::identity]
fn func$0() {
    func();
}
"#,
            expect![[r#"
                func Function FileId(0) 25..50 28..32

                FileId(0) 41..45
            "#]],
        )
    }

    #[test]
    fn attr_assoc_item() {
        check(
            r#"
//- proc_macros: identity

trait Trait {
    #[proc_macros::identity]
    fn func() {
        Self::func$0();
    }
}
"#,
            expect![[r#"
                func Function FileId(0) 48..87 51..55 Trait

                FileId(0) 74..78
            "#]],
        )
    }

    // FIXME: import is classified as function
    #[test]
    fn attr() {
        check(
            r#"
//- proc_macros: identity
use proc_macros::identity;

#[proc_macros::$0identity]
fn func() {}
"#,
            expect![[r#"
                identity Attribute FileId(1) 1..107 32..40

                FileId(0) 43..51
            "#]],
        );
        check(
            r#"
#![crate_type="proc-macro"]
#[proc_macro_attribute]
fn func$0() {}
"#,
            expect![[r#"
                func Attribute FileId(0) 28..64 55..59

                (no references)
            "#]],
        );
    }

    // FIXME: import is classified as function
    #[test]
    fn proc_macro() {
        check(
            r#"
//- proc_macros: mirror
use proc_macros::mirror;

mirror$0! {}
"#,
            expect![[r#"
                mirror Macro FileId(1) 1..77 22..28

                FileId(0) 26..32
            "#]],
        )
    }

    #[test]
    fn derive() {
        check(
            r#"
//- proc_macros: derive_identity
//- minicore: derive
use proc_macros::DeriveIdentity;

#[derive(proc_macros::DeriveIdentity$0)]
struct Foo;
"#,
            expect![[r#"
                derive_identity Derive FileId(2) 1..107 45..60

                FileId(0) 17..31 Import
                FileId(0) 56..70
            "#]],
        );
        check(
            r#"
#![crate_type="proc-macro"]
#[proc_macro_derive(Derive, attributes(x))]
pub fn deri$0ve(_stream: TokenStream) -> TokenStream {}
"#,
            expect![[r#"
                derive Derive FileId(0) 28..125 79..85

                (no references)
            "#]],
        );
    }

    #[test]
    fn assoc_items_trait_def() {
        check(
            r#"
trait Trait {
    const CONST$0: usize;
}

impl Trait for () {
    const CONST: usize = 0;
}

impl Trait for ((),) {
    const CONST: usize = 0;
}

fn f<T: Trait>() {
    let _ = <()>::CONST;

    let _ = T::CONST;
}
"#,
            expect![[r#"
                CONST Const FileId(0) 18..37 24..29 Trait

                FileId(0) 71..76
                FileId(0) 125..130
                FileId(0) 183..188
                FileId(0) 206..211
            "#]],
        );
        check(
            r#"
trait Trait {
    type TypeAlias$0;
}

impl Trait for () {
    type TypeAlias = ();
}

impl Trait for ((),) {
    type TypeAlias = ();
}

fn f<T: Trait>() {
    let _: <() as Trait>::TypeAlias;

    let _: T::TypeAlias;
}
"#,
            expect![[r#"
                TypeAlias TypeAlias FileId(0) 18..33 23..32 Trait

                FileId(0) 66..75
                FileId(0) 117..126
                FileId(0) 181..190
                FileId(0) 207..216
            "#]],
        );
        check(
            r#"
trait Trait {
    fn function$0() {}
}

impl Trait for () {
    fn function() {}
}

impl Trait for ((),) {
    fn function() {}
}

fn f<T: Trait>() {
    let _ = <()>::function;

    let _ = T::function;
}
"#,
            expect![[r#"
                function Function FileId(0) 18..34 21..29 Trait

                FileId(0) 65..73
                FileId(0) 112..120
                FileId(0) 166..174
                FileId(0) 192..200
            "#]],
        );
    }

    #[test]
    fn assoc_items_trait_impl_def() {
        check(
            r#"
trait Trait {
    const CONST: usize;
}

impl Trait for () {
    const CONST$0: usize = 0;
}

impl Trait for ((),) {
    const CONST: usize = 0;
}

fn f<T: Trait>() {
    let _ = <()>::CONST;

    let _ = T::CONST;
}
"#,
            expect![[r#"
                CONST Const FileId(0) 65..88 71..76

                FileId(0) 183..188
            "#]],
        );
        check(
            r#"
trait Trait {
    type TypeAlias;
}

impl Trait for () {
    type TypeAlias$0 = ();
}

impl Trait for ((),) {
    type TypeAlias = ();
}

fn f<T: Trait>() {
    let _: <() as Trait>::TypeAlias;

    let _: T::TypeAlias;
}
"#,
            expect![[r#"
                TypeAlias TypeAlias FileId(0) 61..81 66..75

                FileId(0) 23..32
                FileId(0) 117..126
                FileId(0) 181..190
                FileId(0) 207..216
            "#]],
        );
        check(
            r#"
trait Trait {
    fn function() {}
}

impl Trait for () {
    fn function$0() {}
}

impl Trait for ((),) {
    fn function() {}
}

fn f<T: Trait>() {
    let _ = <()>::function;

    let _ = T::function;
}
"#,
            expect![[r#"
                function Function FileId(0) 62..78 65..73

                FileId(0) 166..174
            "#]],
        );
    }

    #[test]
    fn assoc_items_ref() {
        check(
            r#"
trait Trait {
    const CONST: usize;
}

impl Trait for () {
    const CONST: usize = 0;
}

impl Trait for ((),) {
    const CONST: usize = 0;
}

fn f<T: Trait>() {
    let _ = <()>::CONST$0;

    let _ = T::CONST;
}
"#,
            expect![[r#"
                CONST Const FileId(0) 65..88 71..76

                FileId(0) 183..188
            "#]],
        );
        check(
            r#"
trait Trait {
    type TypeAlias;
}

impl Trait for () {
    type TypeAlias = ();
}

impl Trait for ((),) {
    type TypeAlias = ();
}

fn f<T: Trait>() {
    let _: <() as Trait>::TypeAlias$0;

    let _: T::TypeAlias;
}
"#,
            expect![[r#"
                TypeAlias TypeAlias FileId(0) 18..33 23..32 Trait

                FileId(0) 66..75
                FileId(0) 117..126
                FileId(0) 181..190
                FileId(0) 207..216
            "#]],
        );
        check(
            r#"
trait Trait {
    fn function() {}
}

impl Trait for () {
    fn function() {}
}

impl Trait for ((),) {
    fn function() {}
}

fn f<T: Trait>() {
    let _ = <()>::function$0;

    let _ = T::function;
}
"#,
            expect![[r#"
                function Function FileId(0) 62..78 65..73

                FileId(0) 166..174
            "#]],
        );
    }

    #[test]
    fn name_clashes() {
        check(
            r#"
trait Foo {
    fn method$0(&self) -> u8;
}

struct Bar {
    method: u8,
}

impl Foo for Bar {
    fn method(&self) -> u8 {
        self.method
    }
}
fn method() {}
"#,
            expect![[r#"
                method Function FileId(0) 16..39 19..25 Foo

                FileId(0) 101..107
            "#]],
        );
        check(
            r#"
trait Foo {
    fn method(&self) -> u8;
}

struct Bar {
    method$0: u8,
}

impl Foo for Bar {
    fn method(&self) -> u8 {
        self.method
    }
}
fn method() {}
"#,
            expect![[r#"
                method Field FileId(0) 60..70 60..66

                FileId(0) 136..142 Read
            "#]],
        );
        check(
            r#"
trait Foo {
    fn method(&self) -> u8;
}

struct Bar {
    method: u8,
}

impl Foo for Bar {
    fn method$0(&self) -> u8 {
        self.method
    }
}
fn method() {}
"#,
            expect![[r#"
                method Function FileId(0) 98..148 101..107

                (no references)
            "#]],
        );
        check(
            r#"
trait Foo {
    fn method(&self) -> u8;
}

struct Bar {
    method: u8,
}

impl Foo for Bar {
    fn method(&self) -> u8 {
        self.method$0
    }
}
fn method() {}
"#,
            expect![[r#"
                method Field FileId(0) 60..70 60..66

                FileId(0) 136..142 Read
            "#]],
        );
        check(
            r#"
trait Foo {
    fn method(&self) -> u8;
}

struct Bar {
    method: u8,
}

impl Foo for Bar {
    fn method(&self) -> u8 {
        self.method
    }
}
fn method$0() {}
"#,
            expect![[r#"
                method Function FileId(0) 151..165 154..160

                (no references)
            "#]],
        );
    }

    #[test]
    fn raw_identifier() {
        check(
            r#"
fn r#fn$0() {}
fn main() { r#fn(); }
"#,
            expect![[r#"
                r#fn Function FileId(0) 0..12 3..7

                FileId(0) 25..29
            "#]],
        );
    }
}
