//! This module implements a reference search.
//! First, the element at the cursor position must be either an `ast::Name`
//! or `ast::NameRef`. If it's a `ast::NameRef`, at the classification step we
//! try to resolve the direct tree parent of this element, otherwise we
//! already have a definition and just need to get its HIR together with
//! some information that is needed for futher steps of searching.
//! After that, we collect files that might contain references and look
//! for text occurrences of the identifier. If there's an `ast::NameRef`
//! at the index that the match starts at and its tree parent is
//! resolved to the search element definition, we get a reference.

pub(crate) mod rename;

use hir::Semantics;
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    search::Reference,
    search::{ReferenceAccess, ReferenceKind, SearchScope},
    RootDatabase,
};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, NameOwner},
    match_ast, AstNode, SyntaxKind, SyntaxNode, TextRange, TokenAtOffset,
};

use crate::{display::TryToNav, FilePosition, FileRange, NavigationTarget, RangeInfo};

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    declaration: Declaration,
    references: Vec<Reference>,
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub nav: NavigationTarget,
    pub kind: ReferenceKind,
    pub access: Option<ReferenceAccess>,
}

impl ReferenceSearchResult {
    pub fn declaration(&self) -> &Declaration {
        &self.declaration
    }

    pub fn decl_target(&self) -> &NavigationTarget {
        &self.declaration.nav
    }

    pub fn references(&self) -> &[Reference] {
        &self.references
    }

    /// Total number of references
    /// At least 1 since all valid references should
    /// Have a declaration
    pub fn len(&self) -> usize {
        self.references.len() + 1
    }
}

// allow turning ReferenceSearchResult into an iterator
// over References
impl IntoIterator for ReferenceSearchResult {
    type Item = Reference;
    type IntoIter = std::vec::IntoIter<Reference>;

    fn into_iter(mut self) -> Self::IntoIter {
        let mut v = Vec::with_capacity(self.len());
        v.push(Reference {
            file_range: FileRange {
                file_id: self.declaration.nav.file_id,
                range: self.declaration.nav.focus_or_full_range(),
            },
            kind: self.declaration.kind,
            access: self.declaration.access,
        });
        v.append(&mut self.references);
        v.into_iter()
    }
}

pub(crate) fn find_all_refs(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Option<RangeInfo<ReferenceSearchResult>> {
    let _p = profile::span("find_all_refs");
    let syntax = sema.parse(position.file_id).syntax().clone();

    if let Some(res) = try_find_self_references(&syntax, position) {
        return Some(res);
    }

    let (opt_name, search_kind) = if let Some(name) =
        get_struct_def_name_for_struct_literal_search(&sema, &syntax, position)
    {
        (Some(name), ReferenceKind::StructLiteral)
    } else if let Some(name) = get_enum_def_name_for_struct_literal_search(&sema, &syntax, position)
    {
        (Some(name), ReferenceKind::EnumLiteral)
    } else {
        (
            sema.find_node_at_offset_with_descend::<ast::Name>(&syntax, position.offset),
            ReferenceKind::Other,
        )
    };

    let RangeInfo { range, info: def } = find_name(&sema, &syntax, position, opt_name)?;

    let references = def
        .usages(sema)
        .set_scope(search_scope)
        .all()
        .into_iter()
        .filter(|r| search_kind == ReferenceKind::Other || search_kind == r.kind)
        .collect();

    let nav = def.try_to_nav(sema.db)?;
    let decl_range = nav.focus_or_full_range();

    let mut kind = ReferenceKind::Other;
    if let Definition::Local(local) = def {
        if let either::Either::Left(pat) = local.source(sema.db).value {
            if matches!(
                pat.syntax().parent().and_then(ast::RecordPatField::cast),
                Some(pat_field) if pat_field.name_ref().is_none()
            ) {
                kind = ReferenceKind::FieldShorthandForLocal;
            }
        }
    };

    let declaration = Declaration { nav, kind, access: decl_access(&def, &syntax, decl_range) };

    Some(RangeInfo::new(range, ReferenceSearchResult { declaration, references }))
}

fn find_name(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
    opt_name: Option<ast::Name>,
) -> Option<RangeInfo<Definition>> {
    if let Some(name) = opt_name {
        let def = NameClass::classify(sema, &name)?.referenced_or_defined(sema.db);
        let range = name.syntax().text_range();
        return Some(RangeInfo::new(range, def));
    }
    let name_ref =
        sema.find_node_at_offset_with_descend::<ast::NameRef>(&syntax, position.offset)?;
    let def = NameRefClass::classify(sema, &name_ref)?.referenced(sema.db);
    let range = name_ref.syntax().text_range();
    Some(RangeInfo::new(range, def))
}

fn decl_access(def: &Definition, syntax: &SyntaxNode, range: TextRange) -> Option<ReferenceAccess> {
    match def {
        Definition::Local(_) | Definition::Field(_) => {}
        _ => return None,
    };

    let stmt = find_node_at_offset::<ast::LetStmt>(syntax, range.start())?;
    if stmt.initializer().is_some() {
        let pat = stmt.pat()?;
        if let ast::Pat::IdentPat(it) = pat {
            if it.mut_token().is_some() {
                return Some(ReferenceAccess::Write);
            }
        }
    }

    None
}

fn get_struct_def_name_for_struct_literal_search(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<ast::Name> {
    if let TokenAtOffset::Between(ref left, ref right) = syntax.token_at_offset(position.offset) {
        if right.kind() != SyntaxKind::L_CURLY && right.kind() != SyntaxKind::L_PAREN {
            return None;
        }
        if let Some(name) =
            sema.find_node_at_offset_with_descend::<ast::Name>(&syntax, left.text_range().start())
        {
            return name.syntax().ancestors().find_map(ast::Struct::cast).and_then(|l| l.name());
        }
        if sema
            .find_node_at_offset_with_descend::<ast::GenericParamList>(
                &syntax,
                left.text_range().start(),
            )
            .is_some()
        {
            return left.ancestors().find_map(ast::Struct::cast).and_then(|l| l.name());
        }
    }
    None
}

fn get_enum_def_name_for_struct_literal_search(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<ast::Name> {
    if let TokenAtOffset::Between(ref left, ref right) = syntax.token_at_offset(position.offset) {
        if right.kind() != SyntaxKind::L_CURLY && right.kind() != SyntaxKind::L_PAREN {
            return None;
        }
        if let Some(name) =
            sema.find_node_at_offset_with_descend::<ast::Name>(&syntax, left.text_range().start())
        {
            return name.syntax().ancestors().find_map(ast::Enum::cast).and_then(|l| l.name());
        }
        if sema
            .find_node_at_offset_with_descend::<ast::GenericParamList>(
                &syntax,
                left.text_range().start(),
            )
            .is_some()
        {
            return left.ancestors().find_map(ast::Enum::cast).and_then(|l| l.name());
        }
    }
    None
}

fn try_find_self_references(
    syntax: &SyntaxNode,
    position: FilePosition,
) -> Option<RangeInfo<ReferenceSearchResult>> {
    let self_token =
        syntax.token_at_offset(position.offset).find(|t| t.kind() == SyntaxKind::SELF_KW)?;
    let parent = self_token.parent();
    match_ast! {
        match parent {
            ast::SelfParam(it) => (),
            ast::PathSegment(segment) => {
                segment.self_token()?;
                let path = segment.parent_path();
                if path.qualifier().is_some() && !ast::PathExpr::can_cast(path.syntax().parent()?.kind()) {
                    return None;
                }
            },
            _ => return None,
        }
    };
    let function = parent.ancestors().find_map(ast::Fn::cast)?;
    let self_param = function.param_list()?.self_param()?;
    let param_self_token = self_param.self_token()?;

    let declaration = Declaration {
        nav: NavigationTarget {
            file_id: position.file_id,
            full_range: self_param.syntax().text_range(),
            focus_range: Some(param_self_token.text_range()),
            name: param_self_token.text().clone(),
            kind: param_self_token.kind(),
            container_name: None,
            description: None,
            docs: None,
        },
        kind: ReferenceKind::SelfKw,
        access: Some(if self_param.mut_token().is_some() {
            ReferenceAccess::Write
        } else {
            ReferenceAccess::Read
        }),
    };
    let references = function
        .body()
        .map(|body| {
            body.syntax()
                .descendants()
                .filter_map(ast::PathExpr::cast)
                .filter_map(|expr| {
                    let path = expr.path()?;
                    if path.qualifier().is_none() {
                        path.segment()?.self_token()
                    } else {
                        None
                    }
                })
                .map(|token| Reference {
                    file_range: FileRange { file_id: position.file_id, range: token.text_range() },
                    kind: ReferenceKind::SelfKw,
                    access: declaration.access, // FIXME: properly check access kind here instead of copying it from the declaration
                })
                .collect()
        })
        .unwrap_or_default();

    Some(RangeInfo::new(
        param_self_token.text_range(),
        ReferenceSearchResult { declaration, references },
    ))
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use ide_db::base_db::FileId;
    use stdx::format_to;

    use crate::{fixture, SearchScope};

    #[test]
    fn test_struct_literal_after_space() {
        check(
            r#"
struct Foo <|>{
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
                Foo STRUCT FileId(0) 0..26 7..10 Other

                FileId(0) 101..104 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_before_space() {
        check(
            r#"
struct Foo<|> {}
    fn main() {
    let f: Foo;
    f = Foo {};
}
"#,
            expect![[r#"
                Foo STRUCT FileId(0) 0..13 7..10 Other

                FileId(0) 41..44 Other
                FileId(0) 54..57 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_with_generic_type() {
        check(
            r#"
struct Foo<T> <|>{}
    fn main() {
    let f: Foo::<i32>;
    f = Foo {};
}
"#,
            expect![[r#"
                Foo STRUCT FileId(0) 0..16 7..10 Other

                FileId(0) 64..67 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_struct_literal_for_tuple() {
        check(
            r#"
struct Foo<|>(i32);

fn main() {
    let f: Foo;
    f = Foo(1);
}
"#,
            expect![[r#"
                Foo STRUCT FileId(0) 0..16 7..10 Other

                FileId(0) 54..57 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_enum_after_space() {
        check(
            r#"
enum Foo <|>{
    A,
    B,
}
fn main() {
    let f: Foo;
    f = Foo::A;
}
"#,
            expect![[r#"
                Foo ENUM FileId(0) 0..26 5..8 Other

                FileId(0) 63..66 EnumLiteral
            "#]],
        );
    }

    #[test]
    fn test_enum_before_space() {
        check(
            r#"
enum Foo<|> {
    A,
    B,
}
fn main() {
    let f: Foo;
    f = Foo::A;
}
"#,
            expect![[r#"
                Foo ENUM FileId(0) 0..26 5..8 Other

                FileId(0) 50..53 Other
                FileId(0) 63..66 EnumLiteral
            "#]],
        );
    }

    #[test]
    fn test_enum_with_generic_type() {
        check(
            r#"
enum Foo<T> <|>{
    A(T),
    B,
}
fn main() {
    let f: Foo<i8>;
    f = Foo::A(1);
}
"#,
            expect![[r#"
                Foo ENUM FileId(0) 0..32 5..8 Other

                FileId(0) 73..76 EnumLiteral
            "#]],
        );
    }

    #[test]
    fn test_enum_for_tuple() {
        check(
            r#"
enum Foo<|>{
    A(i8),
    B(i8),
}
fn main() {
    let f: Foo;
    f = Foo::A(1);
}
"#,
            expect![[r#"
                Foo ENUM FileId(0) 0..33 5..8 Other

                FileId(0) 70..73 EnumLiteral
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
    i = i<|> + j;

    {
        i = 0;
    }

    i = 5;
}"#,
            expect![[r#"
                i IDENT_PAT FileId(0) 24..25 Other Write

                FileId(0) 50..51 Other Write
                FileId(0) 54..55 Other Read
                FileId(0) 76..77 Other Write
                FileId(0) 94..95 Other Write
            "#]],
        );
    }

    #[test]
    fn search_filters_by_range() {
        check(
            r#"
fn foo() {
    let spam<|> = 92;
    spam + spam
}
fn bar() {
    let spam = 92;
    spam + spam
}
"#,
            expect![[r#"
                spam IDENT_PAT FileId(0) 19..23 Other

                FileId(0) 34..38 Other Read
                FileId(0) 41..45 Other Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_for_param_inside() {
        check(
            r#"
fn foo(i : u32) -> u32 { i<|> }
"#,
            expect![[r#"
                i IDENT_PAT FileId(0) 7..8 Other

                FileId(0) 25..26 Other Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_for_fn_param() {
        check(
            r#"
fn foo(i<|> : u32) -> u32 { i }
"#,
            expect![[r#"
                i IDENT_PAT FileId(0) 7..8 Other

                FileId(0) 25..26 Other Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_field_name() {
        check(
            r#"
//- /lib.rs
struct Foo {
    pub spam<|>: u32,
}

fn main(s: Foo) {
    let f = s.spam;
}
"#,
            expect![[r#"
                spam RECORD_FIELD FileId(0) 17..30 21..25 Other

                FileId(0) 67..71 Other Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_impl_item_name() {
        check(
            r#"
struct Foo;
impl Foo {
    fn f<|>(&self) {  }
}
"#,
            expect![[r#"
                f FN FileId(0) 27..43 30..31 Other

            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_name() {
        check(
            r#"
enum Foo {
    A,
    B<|>,
    C,
}
"#,
            expect![[r#"
                B VARIANT FileId(0) 22..23 22..23 Other

            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_field() {
        check(
            r#"
enum Foo {
    A,
    B { field<|>: u8 },
    C,
}
"#,
            expect![[r#"
                field RECORD_FIELD FileId(0) 26..35 26..31 Other

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
    let i = foo::Foo<|> { n: 5 };
}
"#,
            expect![[r#"
                Foo STRUCT FileId(1) 17..51 28..31 Other

                FileId(0) 53..56 StructLiteral
                FileId(2) 79..82 StructLiteral
            "#]],
        );
    }

    // `mod foo;` is not in the results because `foo` is an `ast::Name`.
    // So, there are two references: the first one is a definition of the `foo` module,
    // which is the whole `foo.rs`, and the second one is in `use foo::Foo`.
    #[test]
    fn test_find_all_refs_decl_module() {
        check(
            r#"
//- /lib.rs
mod foo<|>;

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
                foo SOURCE_FILE FileId(1) 0..35 Other

                FileId(0) 14..17 Other
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
pub(super) struct Foo<|> {
    pub n: u32,
}
"#,
            expect![[r#"
                Foo STRUCT FileId(2) 0..41 18..21 Other

                FileId(1) 20..23 Other
                FileId(1) 47..50 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_with_scope() {
        let code = r#"
            //- /lib.rs
            mod foo;
            mod bar;

            pub fn quux<|>() {}

            //- /foo.rs
            fn f() { super::quux(); }

            //- /bar.rs
            fn f() { super::quux(); }
        "#;

        check_with_scope(
            code,
            None,
            expect![[r#"
                quux FN FileId(0) 19..35 26..30 Other

                FileId(1) 16..20 StructLiteral
                FileId(2) 16..20 StructLiteral
            "#]],
        );

        check_with_scope(
            code,
            Some(SearchScope::single_file(FileId(2))),
            expect![[r#"
                quux FN FileId(0) 19..35 26..30 Other

                FileId(2) 16..20 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_macro_def() {
        check(
            r#"
#[macro_export]
macro_rules! m1<|> { () => (()) }

fn foo() {
    m1();
    m1();
}
"#,
            expect![[r#"
                m1 MACRO_RULES FileId(0) 0..46 29..31 Other

                FileId(0) 63..65 StructLiteral
                FileId(0) 73..75 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_basic_highlight_read_write() {
        check(
            r#"
fn foo() {
    let mut i<|> = 0;
    i = i + 1;
}
"#,
            expect![[r#"
                i IDENT_PAT FileId(0) 23..24 Other Write

                FileId(0) 34..35 Other Write
                FileId(0) 38..39 Other Read
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
    s.f<|> = 0;
}
"#,
            expect![[r#"
                f RECORD_FIELD FileId(0) 15..21 15..16 Other

                FileId(0) 55..56 RecordFieldExprOrPat Read
                FileId(0) 68..69 Other Write
            "#]],
        );
    }

    #[test]
    fn test_basic_highlight_decl_no_write() {
        check(
            r#"
fn foo() {
    let i<|>;
    i = 1;
}
"#,
            expect![[r#"
                i IDENT_PAT FileId(0) 19..20 Other

                FileId(0) 26..27 Other Write
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
        pub fn new<|>() -> Foo { Foo }
    }
}

fn main() {
    let _f = foo::Foo::new();
}
"#,
            expect![[r#"
                new FN FileId(0) 54..81 61..64 Other

                FileId(0) 126..129 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_nested_module() {
        check(
            r#"
//- /lib.rs
mod foo { mod bar; }

fn f<|>() {}

//- /foo/bar.rs
use crate::f;

fn g() { f(); }
"#,
            expect![[r#"
                f FN FileId(0) 22..31 25..26 Other

                FileId(1) 11..12 Other
                FileId(1) 24..25 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_struct_pat() {
        check(
            r#"
struct S {
    field<|>: u8,
}

fn f(s: S) {
    match s {
        S { field } => {}
    }
}
"#,
            expect![[r#"
                field RECORD_FIELD FileId(0) 15..24 15..20 Other

                FileId(0) 68..73 FieldShorthandForField Read
            "#]],
        );
    }

    #[test]
    fn test_find_all_refs_enum_var_pat() {
        check(
            r#"
enum En {
    Variant {
        field<|>: u8,
    }
}

fn f(e: En) {
    match e {
        En::Variant { field } => {}
    }
}
"#,
            expect![[r#"
                field RECORD_FIELD FileId(0) 32..41 32..37 Other

                FileId(0) 102..107 FieldShorthandForField Read
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
            field<|>: u8,
        }
    }
}

fn f() -> m::En {
    m::En::Variant { field: 0 }
}
"#,
            expect![[r#"
                field RECORD_FIELD FileId(0) 56..65 56..61 Other

                FileId(0) 125..130 RecordFieldExprOrPat Read
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
        let x = self<|>.bar;
        if true {
            let _ = match () {
                () => self,
            };
        }
    }
}
"#,
            expect![[r#"
                self SELF_KW FileId(0) 47..51 47..51 SelfKw Read

                FileId(0) 71..75 SelfKw Read
                FileId(0) 152..156 SelfKw Read
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
        {
            let decl = refs.declaration;
            format_to!(actual, "{} {:?}", decl.nav.debug_render(), decl.kind);
            if let Some(access) = decl.access {
                format_to!(actual, " {:?}", access)
            }
            actual += "\n\n";
        }

        for r in &refs.references {
            format_to!(actual, "{:?} {:?} {:?}", r.file_range.file_id, r.file_range.range, r.kind);
            if let Some(access) = r.access {
                format_to!(actual, " {:?}", access);
            }
            actual += "\n";
        }
        expect.assert_eq(&actual)
    }
}
