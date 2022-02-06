use expect_test::{expect, Expect};
use hir::{HasAttrs, Semantics};
use ide_db::{
    base_db::{FilePosition, FileRange},
    defs::Definition,
    RootDatabase,
};
use itertools::Itertools;
use syntax::{ast, match_ast, AstNode, SyntaxNode};

use crate::{
    doc_links::{extract_definitions_from_docs, resolve_doc_path_for_def, rewrite_links},
    fixture, TryToNav,
};

fn check_external_docs(ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let url = analysis.external_docs(position).unwrap().expect("could not find url for symbol");

    expect.assert_eq(&url)
}

fn check_rewrite(ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let sema = &Semantics::new(&*analysis.db);
    let (cursor_def, docs) = def_under_cursor(sema, &position);
    let res = rewrite_links(sema.db, docs.as_str(), cursor_def);
    expect.assert_eq(&res)
}

fn check_doc_links(ra_fixture: &str) {
    let key_fn = |&(FileRange { file_id, range }, _): &_| (file_id, range.start());

    let (analysis, position, mut expected) = fixture::annotations(ra_fixture);
    expected.sort_by_key(key_fn);
    let sema = &Semantics::new(&*analysis.db);
    let (cursor_def, docs) = def_under_cursor(sema, &position);
    let defs = extract_definitions_from_docs(&docs);
    let actual: Vec<_> = defs
        .into_iter()
        .map(|(_, link, ns)| {
            let def = resolve_doc_path_for_def(sema.db, cursor_def, &link, ns)
                .unwrap_or_else(|| panic!("Failed to resolve {}", link));
            let nav_target = def.try_to_nav(sema.db).unwrap();
            let range =
                FileRange { file_id: nav_target.file_id, range: nav_target.focus_or_full_range() };
            (range, link)
        })
        .sorted_by_key(key_fn)
        .collect();
    assert_eq!(expected, actual);
}

fn def_under_cursor(
    sema: &Semantics<RootDatabase>,
    position: &FilePosition,
) -> (Definition, hir::Documentation) {
    let (docs, def) = sema
        .parse(position.file_id)
        .syntax()
        .token_at_offset(position.offset)
        .left_biased()
        .unwrap()
        .ancestors()
        .find_map(|it| node_to_def(sema, &it))
        .expect("no def found")
        .unwrap();
    let docs = docs.expect("no docs found for cursor def");
    (def, docs)
}

fn node_to_def(
    sema: &Semantics<RootDatabase>,
    node: &SyntaxNode,
) -> Option<Option<(Option<hir::Documentation>, Definition)>> {
    Some(match_ast! {
        match node {
            ast::SourceFile(it)  => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Module(def))),
            ast::Module(it)      => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Module(def))),
            ast::Fn(it)          => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Function(def))),
            ast::Struct(it)      => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Adt(hir::Adt::Struct(def)))),
            ast::Union(it)       => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Adt(hir::Adt::Union(def)))),
            ast::Enum(it)        => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Adt(hir::Adt::Enum(def)))),
            ast::Variant(it)     => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Variant(def))),
            ast::Trait(it)       => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Trait(def))),
            ast::Static(it)      => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Static(def))),
            ast::Const(it)       => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Const(def))),
            ast::TypeAlias(it)   => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::TypeAlias(def))),
            ast::Impl(it)        => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::SelfType(def))),
            ast::RecordField(it) => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Field(def))),
            ast::TupleField(it)  => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Field(def))),
            ast::Macro(it)       => sema.to_def(&it).map(|def| (def.docs(sema.db), Definition::Macro(def))),
            // ast::Use(it) => sema.to_def(&it).map(|def| (Box::new(it) as _, def.attrs(sema.db))),
            _ => return None,
        }
    })
}

#[test]
fn external_docs_doc_url_crate() {
    check_external_docs(
        r#"
//- /main.rs crate:main deps:foo
use foo$0::Foo;
//- /lib.rs crate:foo
pub struct Foo;
"#,
        expect![[r#"https://docs.rs/foo/*/foo/index.html"#]],
    );
}

#[test]
fn external_docs_doc_url_std_crate() {
    check_external_docs(
        r#"
//- /main.rs crate:std
use self$0;
"#,
        expect![[r#"https://doc.rust-lang.org/nightly/std/index.html"#]],
    );
}

#[test]
fn external_docs_doc_url_struct() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Fo$0o;
"#,
        expect![[r#"https://docs.rs/foo/*/foo/struct.Foo.html"#]],
    );
}

#[test]
fn external_docs_doc_url_struct_field() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo {
    field$0: ()
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#structfield.field"##]],
    );
}

#[test]
fn external_docs_doc_url_fn() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub fn fo$0o() {}
"#,
        expect![[r#"https://docs.rs/foo/*/foo/fn.foo.html"#]],
    );
}

#[test]
fn external_docs_doc_url_impl_assoc() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo;
impl Foo {
    pub fn method$0() {}
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#method.method"##]],
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo;
impl Foo {
    const CONST$0: () = ();
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#associatedconstant.CONST"##]],
    );
}

#[test]
fn external_docs_doc_url_impl_trait_assoc() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo;
pub trait Trait {
    fn method() {}
}
impl Trait for Foo {
    pub fn method$0() {}
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#method.method"##]],
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo;
pub trait Trait {
    const CONST: () = ();
}
impl Trait for Foo {
    const CONST$0: () = ();
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#associatedconstant.CONST"##]],
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo;
pub trait Trait {
    type Type;
}
impl Trait for Foo {
    type Type$0 = ();
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#associatedtype.Type"##]],
    );
}

#[test]
fn external_docs_doc_url_trait_assoc() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub trait Foo {
    fn method$0();
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/trait.Foo.html#tymethod.method"##]],
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub trait Foo {
    const CONST$0: ();
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/trait.Foo.html#associatedconstant.CONST"##]],
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub trait Foo {
    type Type$0;
}
"#,
        expect![[r##"https://docs.rs/foo/*/foo/trait.Foo.html#associatedtype.Type"##]],
    );
}

#[test]
fn external_docs_trait() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
trait Trait$0 {}
"#,
        expect![[r#"https://docs.rs/foo/*/foo/trait.Trait.html"#]],
    )
}

#[test]
fn external_docs_module() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub mod foo {
    pub mod ba$0r {}
}
"#,
        expect![[r#"https://docs.rs/foo/*/foo/foo/bar/index.html"#]],
    )
}

#[test]
fn external_docs_reexport_order() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub mod wrapper {
    pub use module::Item;

    pub mod module {
        pub struct Item;
    }
}

fn foo() {
    let bar: wrapper::It$0em;
}
        "#,
        expect![[r#"https://docs.rs/foo/*/foo/wrapper/module/struct.Item.html"#]],
    )
}

#[test]
fn test_trait_items() {
    check_doc_links(
        r#"
/// [`Trait`]
/// [`Trait::Type`]
/// [`Trait::CONST`]
/// [`Trait::func`]
trait Trait$0 {
   // ^^^^^ Trait
    type Type;
      // ^^^^ Trait::Type
    const CONST: usize;
       // ^^^^^ Trait::CONST
    fn func();
    // ^^^^ Trait::func
}
        "#,
    )
}

#[test]
fn rewrite_html_root_url() {
    check_rewrite(
        r#"
//- /main.rs crate:foo
#![doc(arbitrary_attribute = "test", html_root_url = "https:/example.com", arbitrary_attribute2)]

pub mod foo {
    pub struct Foo;
}
/// [Foo](foo::Foo)
pub struct B$0ar
"#,
        expect![[r#"[Foo](https://example.com/foo/foo/struct.Foo.html)"#]],
    );
}

#[test]
fn rewrite_on_field() {
    check_rewrite(
        r#"
//- /main.rs crate:foo
pub struct Foo {
    /// [Foo](struct.Foo.html)
    fie$0ld: ()
}
"#,
        expect![[r#"[Foo](https://docs.rs/foo/*/foo/struct.Foo.html)"#]],
    );
}

#[test]
fn rewrite_struct() {
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo]
pub struct $0Foo;
"#,
        expect![[r#"[Foo](https://docs.rs/foo/*/foo/struct.Foo.html)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [`Foo`]
pub struct $0Foo;
"#,
        expect![[r#"[`Foo`](https://docs.rs/foo/*/foo/struct.Foo.html)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo](struct.Foo.html)
pub struct $0Foo;
"#,
        expect![[r#"[Foo](https://docs.rs/foo/*/foo/struct.Foo.html)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [struct Foo](struct.Foo.html)
pub struct $0Foo;
"#,
        expect![[r#"[struct Foo](https://docs.rs/foo/*/foo/struct.Foo.html)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [my Foo][foo]
///
/// [foo]: Foo
pub struct $0Foo;
"#,
        expect![[r#"[my Foo](https://docs.rs/foo/*/foo/struct.Foo.html)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [`foo`]
///
/// [`foo`]: Foo
pub struct $0Foo;
"#,
        expect![["[`foo`]"]],
    );
}
