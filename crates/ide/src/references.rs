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

use hir::Semantics;
use ide_db::{
    base_db::FileId,
    defs::{Definition, NameClass, NameRefClass},
    search::{ReferenceAccess, SearchScope},
    RootDatabase,
};
use rustc_hash::FxHashMap;
use syntax::{
    algo::find_node_at_offset,
    ast::{self, NameOwner},
    AstNode, SyntaxNode, TextRange, TokenAtOffset, T,
};

use crate::{display::TryToNav, FilePosition, NavigationTarget};

#[derive(Debug, Clone)]
pub struct ReferenceSearchResult {
    pub declaration: Declaration,
    pub references: FxHashMap<FileId, Vec<(TextRange, Option<ReferenceAccess>)>>,
}

#[derive(Debug, Clone)]
pub struct Declaration {
    pub nav: NavigationTarget,
    pub access: Option<ReferenceAccess>,
}

pub(crate) fn find_all_refs(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
    search_scope: Option<SearchScope>,
) -> Option<ReferenceSearchResult> {
    let _p = profile::span("find_all_refs");
    let syntax = sema.parse(position.file_id).syntax().clone();

    let (opt_name, ctor_filter): (_, Option<fn(&_) -> bool>) = if let Some(name) =
        get_struct_def_name_for_struct_literal_search(&sema, &syntax, position)
    {
        (
            Some(name),
            Some(|name_ref| is_record_lit_name_ref(name_ref) || is_call_expr_name_ref(name_ref)),
        )
    } else if let Some(name) = get_enum_def_name_for_struct_literal_search(&sema, &syntax, position)
    {
        (Some(name), Some(is_enum_lit_name_ref))
    } else {
        (sema.find_node_at_offset_with_descend::<ast::Name>(&syntax, position.offset), None)
    };

    let def = find_def(&sema, &syntax, position, opt_name)?;

    let mut usages = def.usages(sema).set_scope(search_scope).all();
    if let Some(ctor_filter) = ctor_filter {
        // filter for constructor-literals
        usages.references.iter_mut().for_each(|(&file_id, it)| {
            let root = sema.parse(file_id);
            let root = root.syntax();
            it.retain(|reference| {
                reference.as_name_ref(root).map_or(false, |name_ref| ctor_filter(&name_ref))
            })
        });
        usages.references.retain(|_, it| !it.is_empty());
    }
    let nav = def.try_to_nav(sema.db)?;
    let decl_range = nav.focus_or_full_range();

    let declaration = Declaration { nav, access: decl_access(&def, &syntax, decl_range) };
    let references = usages
        .into_iter()
        .map(|(file_id, refs)| {
            (file_id, refs.into_iter().map(|file_ref| (file_ref.range, file_ref.access)).collect())
        })
        .collect();

    Some(ReferenceSearchResult { declaration, references })
}

fn find_def(
    sema: &Semantics<RootDatabase>,
    syntax: &SyntaxNode,
    position: FilePosition,
    opt_name: Option<ast::Name>,
) -> Option<Definition> {
    if let Some(name) = opt_name {
        let class = NameClass::classify(sema, &name)?;
        Some(class.referenced_or_defined(sema.db))
    } else if let Some(lifetime) =
        sema.find_node_at_offset_with_descend::<ast::Lifetime>(&syntax, position.offset)
    {
        let def = if let Some(def) =
            NameRefClass::classify_lifetime(sema, &lifetime).map(|class| class.referenced(sema.db))
        {
            def
        } else {
            NameClass::classify_lifetime(sema, &lifetime)?.referenced_or_defined(sema.db)
        };
        Some(def)
    } else if let Some(name_ref) =
        sema.find_node_at_offset_with_descend::<ast::NameRef>(&syntax, position.offset)
    {
        let class = NameRefClass::classify(sema, &name_ref)?;
        Some(class.referenced(sema.db))
    } else {
        None
    }
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

fn is_call_expr_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::CallExpr::cast)
        .and_then(|c| match c.expr()? {
            ast::Expr::PathExpr(p) => {
                Some(p.path()?.segment()?.name_ref().as_ref() == Some(name_ref))
            }
            _ => None,
        })
        .unwrap_or(false)
}

fn is_record_lit_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::RecordExpr::cast)
        .and_then(|l| l.path())
        .and_then(|p| p.segment())
        .map(|p| p.name_ref().as_ref() == Some(name_ref))
        .unwrap_or(false)
}

fn is_enum_lit_name_ref(name_ref: &ast::NameRef) -> bool {
    name_ref
        .syntax()
        .ancestors()
        .find_map(ast::PathExpr::cast)
        .and_then(|p| p.path())
        .and_then(|p| p.qualifier())
        .and_then(|p| p.segment())
        .map(|p| p.name_ref().as_ref() == Some(name_ref))
        .unwrap_or(false)
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
                Foo Enum FileId(0) 0..26 5..8

                FileId(0) 63..66
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
                Foo Struct FileId(1) 17..51 28..31

                FileId(0) 53..56
                FileId(2) 79..82
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
                foo Module FileId(1) 0..35

                FileId(0) 14..17
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
                Foo Struct FileId(2) 0..41 18..21

                FileId(1) 20..23
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

                FileId(1) 11..12
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
        {
            let decl = refs.declaration;
            format_to!(actual, "{}", decl.nav.debug_render());
            if let Some(access) = decl.access {
                format_to!(actual, " {:?}", access)
            }
            actual += "\n\n";
        }

        for (file_id, references) in refs.references {
            for (range, access) in references {
                format_to!(actual, "{:?} {:?}", file_id, range);
                if let Some(access) = access {
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
    fn test_find_self_ty_in_trait_def() {
        check(
            r#"
trait Foo {
    fn f() -> Self$0;
}
"#,
            expect![[r#"
                Self TypeParam FileId(0) 6..9 6..9

                FileId(0) 26..30
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
                Bar Variant FileId(0) 11..16 11..14 Other

                FileId(0) 89..92 Other
            "#]],
        );
    }
}
