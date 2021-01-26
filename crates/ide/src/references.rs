//! This module implements a reference search.
//! First, the element at the cursor position must be either an `ast::Name`
//! or `ast::NameRef`. If it's a `ast::NameRef`, at the classification step we
//! try to resolve the direct tree parent of this element, otherwise we
//! already have a definition and just need to get its HIR together with
//! some information that is needed for further steps of searching.
//! After that, we collect files that might contain references and look
//! for text occurrences of the identifier. If there's an `ast::NameRef`
//! at the index that the match starts at and its tree parent is
//! resolved to the search element definition, we get a reference.

pub(crate) mod rename;

use either::Either;
use hir::Semantics;
use ide_db::{
    base_db::FileId,
    defs::{Definition, NameClass, NameRefClass},
    search::{FileReference, ReferenceAccess, ReferenceKind, SearchScope, UsageSearchResult},
    RootDatabase,
};
use syntax::{
    algo::find_node_at_offset,
    ast::{self, NameOwner},
    AstNode, SyntaxNode, TextRange, TokenAtOffset, T,
};

use crate::{display::TryToNav, FilePosition, NavigationTarget};

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    declaration: Declaration,
    references: UsageSearchResult,
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub nav: NavigationTarget,
    pub kind: ReferenceKind,
    pub access: Option<ReferenceAccess>,
}

impl ReferenceSearchResult {
    pub fn references(&self) -> &UsageSearchResult {
        &self.references
    }

    pub fn references_with_declaration(mut self) -> UsageSearchResult {
        let decl_ref = FileReference {
            range: self.declaration.nav.focus_or_full_range(),
            kind: self.declaration.kind,
            access: self.declaration.access,
        };
        let file_id = self.declaration.nav.file_id;
        self.references.references.entry(file_id).or_default().push(decl_ref);
        self.references
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
    type Item = (FileId, Vec<FileReference>);
    type IntoIter = std::collections::hash_map::IntoIter<FileId, Vec<FileReference>>;

    fn into_iter(self) -> Self::IntoIter {
        self.references_with_declaration().into_iter()
    }
}

pub(crate) fn find_all_refs(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Option<ReferenceSearchResult> {
    let _p = profile::span("find_all_refs");
    let syntax = sema.parse(position.file_id).syntax().clone();

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

    let def = find_name(&sema, &syntax, position, opt_name)?;

    let mut usages = def.usages(sema).set_scope(search_scope).all();
    usages
        .references
        .values_mut()
        .for_each(|it| it.retain(|r| search_kind == ReferenceKind::Other || search_kind == r.kind));
    usages.references.retain(|_, it| !it.is_empty());

    let nav = def.try_to_nav(sema.db)?;
    let decl_range = nav.focus_or_full_range();

    let mut kind = ReferenceKind::Other;
    if let Definition::Local(local) = def {
        match local.source(sema.db).value {
            Either::Left(pat) => {
                if matches!(
                    pat.syntax().parent().and_then(ast::RecordPatField::cast),
                    Some(pat_field) if pat_field.name_ref().is_none()
                ) {
                    kind = ReferenceKind::FieldShorthandForLocal;
                }
            }
            Either::Right(_) => kind = ReferenceKind::SelfParam,
        }
    } else if matches!(
        def,
        Definition::GenericParam(hir::GenericParam::LifetimeParam(_)) | Definition::Label(_)
    ) {
        kind = ReferenceKind::Lifetime;
    };

    let declaration = Declaration { nav, kind, access: decl_access(&def, &syntax, decl_range) };

    Some(ReferenceSearchResult { declaration, references: usages })
}

fn find_name(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
    opt_name: Option<ast::Name>,
) -> Option<Definition> {
    let def = if let Some(name) = opt_name {
        NameClass::classify(sema, &name)?.referenced_or_defined(sema.db)
    } else if let Some(lifetime) =
        sema.find_node_at_offset_with_descend::<ast::Lifetime>(&syntax, position.offset)
    {
        if let Some(def) =
            NameRefClass::classify_lifetime(sema, &lifetime).map(|class| class.referenced(sema.db))
        {
            def
        } else {
            NameClass::classify_lifetime(sema, &lifetime)?.referenced_or_defined(sema.db)
        }
    } else if let Some(name_ref) =
        sema.find_node_at_offset_with_descend::<ast::NameRef>(&syntax, position.offset)
    {
        NameRefClass::classify(sema, &name_ref)?.referenced(sema.db)
    } else {
        return None;
    };
    Some(def)
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
        if right.kind() != T!['{'] && right.kind() != T!['('] {
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
        if right.kind() != T!['{'] && right.kind() != T!['('] {
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
                Foo Struct FileId(0) 0..26 7..10 Other

                FileId(0) 101..104 StructLiteral
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
                Foo Struct FileId(0) 0..13 7..10 Other

                FileId(0) 41..44 Other
                FileId(0) 54..57 StructLiteral
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
                Foo Struct FileId(0) 0..16 7..10 Other

                FileId(0) 64..67 StructLiteral
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
                Foo Struct FileId(0) 0..16 7..10 Other

                FileId(0) 54..57 StructLiteral
            "#]],
        );
    }

    #[test]
    fn test_enum_after_space() {
        check(
            r#"
enum Foo $0{
    A,
    B,
}
fn main() {
    let f: Foo;
    f = Foo::A;
}
"#,
            expect![[r#"
                Foo Enum FileId(0) 0..26 5..8 Other

                FileId(0) 63..66 EnumLiteral
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
                Foo Enum FileId(0) 0..26 5..8 Other

                FileId(0) 50..53 Other
                FileId(0) 63..66 EnumLiteral
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
                Foo Enum FileId(0) 0..32 5..8 Other

                FileId(0) 73..76 EnumLiteral
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
                Foo Enum FileId(0) 0..33 5..8 Other

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
    i = i$0 + j;

    {
        i = 0;
    }

    i = 5;
}"#,
            expect![[r#"
                i Local FileId(0) 20..25 24..25 Other Write

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
    let spam$0 = 92;
    spam + spam
}
fn bar() {
    let spam = 92;
    spam + spam
}
"#,
            expect![[r#"
                spam Local FileId(0) 19..23 19..23 Other

                FileId(0) 34..38 Other Read
                FileId(0) 41..45 Other Read
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
                i ValueParam FileId(0) 7..8 7..8 Other

                FileId(0) 25..26 Other Read
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
                i ValueParam FileId(0) 7..8 7..8 Other

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
    pub spam$0: u32,
}

fn main(s: Foo) {
    let f = s.spam;
}
"#,
            expect![[r#"
                spam Field FileId(0) 17..30 21..25 Other

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
    fn f$0(&self) {  }
}
"#,
            expect![[r#"
                f Function FileId(0) 27..43 30..31 Other

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
                B Variant FileId(0) 22..23 22..23 Other

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
                field Field FileId(0) 26..35 26..31 Other

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
                Foo Struct FileId(1) 17..51 28..31 Other

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
                foo Module FileId(1) 0..35 Other

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
pub(super) struct Foo$0 {
    pub n: u32,
}
"#,
            expect![[r#"
                Foo Struct FileId(2) 0..41 18..21 Other

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
                quux Function FileId(0) 19..35 26..30 Other

                FileId(1) 16..20 StructLiteral
                FileId(2) 16..20 StructLiteral
            "#]],
        );

        check_with_scope(
            code,
            Some(SearchScope::single_file(FileId(2))),
            expect![[r#"
                quux Function FileId(0) 19..35 26..30 Other

                FileId(2) 16..20 StructLiteral
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
                m1 Macro FileId(0) 0..46 29..31 Other

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
    let mut i$0 = 0;
    i = i + 1;
}
"#,
            expect![[r#"
                i Local FileId(0) 19..24 23..24 Other Write

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
    s.f$0 = 0;
}
"#,
            expect![[r#"
                f Field FileId(0) 15..21 15..16 Other

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
    let i$0;
    i = 1;
}
"#,
            expect![[r#"
                i Local FileId(0) 19..20 19..20 Other

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
        pub fn new$0() -> Foo { Foo }
    }
}

fn main() {
    let _f = foo::Foo::new();
}
"#,
            expect![[r#"
                new Function FileId(0) 54..81 61..64 Other

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

fn f$0() {}

//- /foo/bar.rs
use crate::f;

fn g() { f(); }
"#,
            expect![[r#"
                f Function FileId(0) 22..31 25..26 Other

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
    field$0: u8,
}

fn f(s: S) {
    match s {
        S { field } => {}
    }
}
"#,
            expect![[r#"
                field Field FileId(0) 15..24 15..20 Other

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
                field Field FileId(0) 32..41 32..37 Other

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
            field$0: u8,
        }
    }
}

fn f() -> m::En {
    m::En::Variant { field: 0 }
}
"#,
            expect![[r#"
                field Field FileId(0) 56..65 56..61 Other

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
                self SelfParam FileId(0) 47..51 47..51 SelfParam

                FileId(0) 71..75 Other Read
                FileId(0) 152..156 Other Read
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
                self SelfParam FileId(0) 47..51 47..51 SelfParam

                FileId(0) 63..67 Other Read
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

        for (file_id, references) in refs.references {
            for r in references {
                format_to!(actual, "{:?} {:?} {:?}", file_id, r.range, r.kind);
                if let Some(access) = r.access {
                    format_to!(actual, " {:?}", access);
                }
                actual += "\n";
            }
        }
        expect.assert_eq(&actual)
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
                'a LifetimeParam FileId(0) 55..57 55..57 Lifetime

                FileId(0) 63..65 Lifetime
                FileId(0) 71..73 Lifetime
                FileId(0) 82..84 Lifetime
                FileId(0) 95..97 Lifetime
                FileId(0) 106..108 Lifetime
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
                'a LifetimeParam FileId(0) 9..11 9..11 Lifetime

                FileId(0) 25..27 Lifetime
                FileId(0) 31..33 Lifetime
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
                'a LifetimeParam FileId(0) 47..49 47..49 Lifetime

                FileId(0) 55..57 Lifetime
                FileId(0) 64..66 Lifetime
                FileId(0) 89..91 Lifetime
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
                a Local FileId(0) 59..60 59..60 Other

                FileId(0) 80..81 Other Read
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
                a Local FileId(0) 59..60 59..60 Other

                FileId(0) 80..81 Other Read
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
                'a Label FileId(0) 29..32 29..31 Lifetime

                FileId(0) 80..82 Lifetime
                FileId(0) 108..110 Lifetime
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
                FOO ConstParam FileId(0) 7..23 13..16 Other

                FileId(0) 42..45 Other
            "#]],
        );
    }

    #[test]
    fn test_find_self_ty_in_trait_def() {
        check(
            r#"
trait Foo {
    fn f() -> Self$0;
}
"#,
            expect![[r#"
                Self TypeParam FileId(0) 6..9 6..9 Other

                FileId(0) 26..30 Other
            "#]],
        );
    }
}
