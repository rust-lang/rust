use std::iter;

use expect_test::{Expect, expect};
use hir::Semantics;
use ide_db::{
    FilePosition, FileRange, RootDatabase,
    defs::Definition,
    documentation::{DocsRangeMap, Documentation, HasDocs},
};
use itertools::Itertools;
use syntax::{AstNode, SyntaxNode, ast, match_ast};

use crate::{
    TryToNav,
    doc_links::{extract_definitions_from_docs, resolve_doc_path_for_def, rewrite_links},
    fixture,
};

fn check_external_docs(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    target_dir: Option<&str>,
    expect_web_url: Option<Expect>,
    expect_local_url: Option<Expect>,
    sysroot: Option<&str>,
) {
    let (analysis, position) = fixture::position(ra_fixture);
    let links = analysis.external_docs(position, target_dir, sysroot).unwrap();

    let web_url = links.web_url;
    let local_url = links.local_url;

    match (expect_web_url, web_url) {
        (Some(expect), Some(url)) => expect.assert_eq(&url),
        (None, None) => (),
        _ => panic!("Unexpected web url"),
    }

    match (expect_local_url, local_url) {
        (Some(expect), Some(url)) => expect.assert_eq(&url),
        (None, None) => (),
        _ => panic!("Unexpected local url"),
    }
}

fn check_rewrite(#[rust_analyzer::rust_fixture] ra_fixture: &str, expect: Expect) {
    let (analysis, position) = fixture::position(ra_fixture);
    let sema = &Semantics::new(&analysis.db);
    let (cursor_def, docs, range) = def_under_cursor(sema, &position);
    let res = rewrite_links(sema.db, docs.as_str(), cursor_def, Some(range));
    expect.assert_eq(&res)
}

fn check_doc_links(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let key_fn = |&(FileRange { file_id, range }, _): &_| (file_id, range.start());

    let (analysis, position, mut expected) = fixture::annotations(ra_fixture);
    expected.sort_by_key(key_fn);
    let sema = &Semantics::new(&analysis.db);
    let (cursor_def, docs, range) = def_under_cursor(sema, &position);
    let defs = extract_definitions_from_docs(&docs);
    let actual: Vec<_> = defs
        .into_iter()
        .flat_map(|(text_range, link, ns)| {
            let attr = range.map(text_range);
            let is_inner_attr = attr.map(|(_file, attr)| attr.is_inner_attr()).unwrap_or(false);
            let def = resolve_doc_path_for_def(sema.db, cursor_def, &link, ns, is_inner_attr)
                .unwrap_or_else(|| panic!("Failed to resolve {link}"));
            def.try_to_nav(sema.db).unwrap().into_iter().zip(iter::repeat(link))
        })
        .map(|(nav_target, link)| {
            let range =
                FileRange { file_id: nav_target.file_id, range: nav_target.focus_or_full_range() };
            (range, link)
        })
        .sorted_by_key(key_fn)
        .collect();
    assert_eq!(expected, actual);
}

fn def_under_cursor(
    sema: &Semantics<'_, RootDatabase>,
    position: &FilePosition,
) -> (Definition, Documentation, DocsRangeMap) {
    let (docs, def) = sema
        .parse_guess_edition(position.file_id)
        .syntax()
        .token_at_offset(position.offset)
        .left_biased()
        .unwrap()
        .parent_ancestors()
        .find_map(|it| node_to_def(sema, &it))
        .expect("no def found")
        .unwrap();
    let (docs, range) = docs.expect("no docs found for cursor def");
    (def, docs, range)
}

fn node_to_def(
    sema: &Semantics<'_, RootDatabase>,
    node: &SyntaxNode,
) -> Option<Option<(Option<(Documentation, DocsRangeMap)>, Definition)>> {
    Some(match_ast! {
        match node {
            ast::SourceFile(it)  => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Module(def))),
            ast::Module(it)      => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Module(def))),
            ast::Fn(it)          => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Function(def))),
            ast::Struct(it)      => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Adt(hir::Adt::Struct(def)))),
            ast::Union(it)       => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Adt(hir::Adt::Union(def)))),
            ast::Enum(it)        => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Adt(hir::Adt::Enum(def)))),
            ast::Variant(it)     => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Variant(def))),
            ast::Trait(it)       => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Trait(def))),
            ast::Static(it)      => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Static(def))),
            ast::Const(it)       => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Const(def))),
            ast::TypeAlias(it)   => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::TypeAlias(def))),
            ast::Impl(it)        => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::SelfType(def))),
            ast::RecordField(it) => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Field(def))),
            ast::TupleField(it)  => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Field(def))),
            ast::Macro(it)       => sema.to_def(&it).map(|def| (def.docs_with_rangemap(sema.db), Definition::Macro(def))),
            // ast::Use(it) => sema.to_def(&it).map(|def| (Box::new(it) as _, def.attrs(sema.db))),
            _ => return None,
        }
    })
}

#[test]
fn external_docs_doc_builtin_type() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
let x: u3$02 = 0;
"#,
        Some("/home/user/project"),
        Some(expect![[r#"https://doc.rust-lang.org/nightly/core/primitive.u32.html"#]]),
        Some(expect![[r#"file:///sysroot/share/doc/rust/html/core/primitive.u32.html"#]]),
        Some("/sysroot"),
    );
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
        Some("/home/user/project"),
        Some(expect![[r#"https://docs.rs/foo/*/foo/index.html"#]]),
        Some(expect![[r#"file:///home/user/project/doc/foo/index.html"#]]),
        Some("/sysroot"),
    );
}

#[test]
fn external_docs_doc_url_std_crate() {
    check_external_docs(
        r#"
//- /main.rs crate:std
use self$0;
"#,
        Some("/home/user/project"),
        Some(expect!["https://doc.rust-lang.org/stable/std/index.html"]),
        Some(expect!["file:///sysroot/share/doc/rust/html/std/index.html"]),
        Some("/sysroot"),
    );
}

#[test]
fn external_docs_doc_url_struct() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Fo$0o;
"#,
        Some("/home/user/project"),
        Some(expect![[r#"https://docs.rs/foo/*/foo/struct.Foo.html"#]]),
        Some(expect![[r#"file:///home/user/project/doc/foo/struct.Foo.html"#]]),
        Some("/sysroot"),
    );
}

#[test]
fn external_docs_doc_url_windows_backslash_path() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Fo$0o;
"#,
        Some(r"C:\Users\user\project"),
        Some(expect![[r#"https://docs.rs/foo/*/foo/struct.Foo.html"#]]),
        Some(expect![[r#"file:///C:/Users/user/project/doc/foo/struct.Foo.html"#]]),
        Some("/sysroot"),
    );
}

#[test]
fn external_docs_doc_url_windows_slash_path() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Fo$0o;
"#,
        Some("C:/Users/user/project"),
        Some(expect![[r#"https://docs.rs/foo/*/foo/struct.Foo.html"#]]),
        Some(expect![[r#"file:///C:/Users/user/project/doc/foo/struct.Foo.html"#]]),
        Some("/sysroot"),
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
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#structfield.field"##]]),
        None,
        None,
    );
}

#[test]
fn external_docs_doc_url_fn() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub fn fo$0o() {}
"#,
        None,
        Some(expect![[r#"https://docs.rs/foo/*/foo/fn.foo.html"#]]),
        None,
        None,
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
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#method.method"##]]),
        None,
        None,
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub struct Foo;
impl Foo {
    const CONST$0: () = ();
}
"#,
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#associatedconstant.CONST"##]]),
        None,
        None,
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
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#method.method"##]]),
        None,
        None,
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
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#associatedconstant.CONST"##]]),
        None,
        None,
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
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/struct.Foo.html#associatedtype.Type"##]]),
        None,
        None,
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
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/trait.Foo.html#tymethod.method"##]]),
        None,
        None,
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub trait Foo {
    const CONST$0: ();
}
"#,
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/trait.Foo.html#associatedconstant.CONST"##]]),
        None,
        None,
    );
    check_external_docs(
        r#"
//- /main.rs crate:foo
pub trait Foo {
    type Type$0;
}
"#,
        None,
        Some(expect![[r##"https://docs.rs/foo/*/foo/trait.Foo.html#associatedtype.Type"##]]),
        None,
        None,
    );
}

#[test]
fn external_docs_trait() {
    check_external_docs(
        r#"
//- /main.rs crate:foo
trait Trait$0 {}
"#,
        None,
        Some(expect![[r#"https://docs.rs/foo/*/foo/trait.Trait.html"#]]),
        None,
        None,
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
        None,
        Some(expect![[r#"https://docs.rs/foo/*/foo/foo/bar/index.html"#]]),
        None,
        None,
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
        None,
        Some(expect![[r#"https://docs.rs/foo/*/foo/wrapper/module/struct.Item.html"#]]),
        None,
        None,
    )
}

#[test]
fn doc_links_items_simple() {
    check_doc_links(
        r#"
//- /main.rs crate:main deps:krate
/// [`krate`]
//! [`Trait`]
//! [`function`]
//! [`CONST`]
//! [`STATIC`]
//! [`Struct`]
//! [`Enum`]
//! [`Union`]
//! [`Type`]
//! [`module`]
use self$0;

const CONST: () = ();
   // ^^^^^ CONST
static STATIC: () = ();
    // ^^^^^^ STATIC
trait Trait {
   // ^^^^^ Trait
}
fn function() {}
// ^^^^^^^^ function
struct Struct;
    // ^^^^^^ Struct
enum Enum {}
  // ^^^^ Enum
union Union {__: ()}
   // ^^^^^ Union
type Type = ();
  // ^^^^ Type
mod module {}
 // ^^^^^^ module
//- /krate.rs crate:krate
// empty
//^file krate
"#,
    )
}

#[test]
fn doc_links_inherent_impl_items() {
    check_doc_links(
        r#"
/// [`Struct::CONST`]
/// [`Struct::function`]
struct Struct$0;

impl Struct {
    const CONST: () = ();
       // ^^^^^ Struct::CONST
    fn function() {}
    // ^^^^^^^^ Struct::function
}
"#,
    )
}

#[test]
fn doc_links_trait_impl_items() {
    check_doc_links(
        r#"
trait Trait {
    type Type;
    const CONST: usize;
       // ^^^^^ Struct::CONST
    fn function();
    // ^^^^^^^^ Struct::function
}
// FIXME #9694: [`Struct::Type`]
/// [`Struct::CONST`]
/// [`Struct::function`]
struct Struct$0;

impl Trait for Struct {
    type Type = ();
    const CONST: () = ();
    fn function() {}
}
"#,
    )
}

#[test]
fn doc_links_trait_items() {
    check_doc_links(
        r#"
/// [`Trait`]
/// [`Trait::Type`]
/// [`Trait::CONST`]
/// [`Trait::function`]
trait Trait$0 {
   // ^^^^^ Trait
type Type;
  // ^^^^ Trait::Type
const CONST: usize;
   // ^^^^^ Trait::CONST
fn function();
// ^^^^^^^^ Trait::function
}
    "#,
    )
}

#[test]
fn doc_links_field() {
    check_doc_links(
        r#"
/// [`S::f`]
/// [`S2::f`]
/// [`T::0`]
/// [`U::a`]
/// [`E::A::f`]
/// [`E::B::0`]
struct S$0 {
    f: i32,
  //^ S::f
  //^ S2::f
}
type S2 = S;
struct T(i32);
       //^^^ T::0
union U {
    a: i32,
  //^ U::a
}
enum E {
    A { f: i32 },
      //^ E::A::f
    B(i32),
    //^^^ E::B::0
}
"#,
    );
}

#[test]
fn doc_links_field_via_self() {
    check_doc_links(
        r#"
/// [`Self::f`]
struct S$0 {
    f: i32,
  //^ Self::f
}
"#,
    );
}

#[test]
fn doc_links_tuple_field_via_self() {
    check_doc_links(
        r#"
/// [`Self::0`]
struct S$0(i32);
       //^^^ Self::0
"#,
    );
}

#[test]
fn doc_links_module() {
    check_doc_links(
        r#"
/// [`M`]
/// [`M::f`]
mod M$0 {
  //^ M
  #![doc = "inner_item[`S`]"]

    pub fn f() {}
         //^ M::f
    pub struct S;
             //^ S
}
"#,
    );

    check_doc_links(
        r#"
mod M$0 {
  //^ super::M
    //! [`super::M`]
    //! [`super::M::f`]
    //! [`super::M::S`]
    pub fn f() {}
         //^ super::M::f
    pub struct S;
             //^ super::M::S
}
"#,
    );
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

#[test]
fn rewrite_intra_doc_link() {
    check_rewrite(
        r#"
        //- minicore: eq, derive
        //- /main.rs crate:foo
        //! $0[PartialEq]
        fn main() {}
        "#,
        expect!["[PartialEq](https://doc.rust-lang.org/stable/core/cmp/trait.PartialEq.html)"],
    );
}

#[test]
fn rewrite_intra_doc_link_with_anchor() {
    check_rewrite(
        r#"
        //- minicore: eq, derive
        //- /main.rs crate:foo
        //! $0[PartialEq#derivable]
        fn main() {}
        "#,
        expect![
            "[PartialEq#derivable](https://doc.rust-lang.org/stable/core/cmp/trait.PartialEq.html#derivable)"
        ],
    );
}

#[test]
fn rewrite_module() {
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo]
pub mod $0Foo{
};
"#,
        expect![[r#"[Foo](https://docs.rs/foo/*/foo/Foo/index.html)"#]],
    );

    check_rewrite(
        r#"
//- /main.rs crate:foo
pub mod $0Foo{
    //! [super::Foo]
};
"#,
        expect![[r#"[super::Foo](https://docs.rs/foo/*/foo/Foo/index.html)"#]],
    );
}

#[test]
fn rewrite_intra_doc_link_to_associated_item() {
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::bar]
pub struct $0Foo;

impl Foo {
    fn bar() {}
}
"#,
        expect![[r#"[Foo::bar](https://docs.rs/foo/*/foo/struct.Foo.html#method.bar)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::bar]
pub struct $0Foo {
    bar: ()
}
"#,
        expect![[r#"[Foo::bar](https://docs.rs/foo/*/foo/struct.Foo.html#structfield.bar)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::Bar]
pub enum $0Foo {
    Bar
}
"#,
        expect![[r#"[Foo::Bar](https://docs.rs/foo/*/foo/enum.Foo.html#variant.Bar)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::BAR]
pub struct $0Foo;

impl Foo {
    const BAR: () = ();
}
"#,
        expect![[
            r#"[Foo::BAR](https://docs.rs/foo/*/foo/struct.Foo.html#associatedconstant.BAR)"#
        ]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::bar]
pub trait $0Foo {
    fn bar();
}
"#,
        expect![[r#"[Foo::bar](https://docs.rs/foo/*/foo/trait.Foo.html#tymethod.bar)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::Bar]
pub trait $0Foo {
    type Bar;
}
"#,
        expect![[r#"[Foo::Bar](https://docs.rs/foo/*/foo/trait.Foo.html#associatedtype.Bar)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [Foo::bar#anchor]
pub struct $0Foo {
    bar: (),
}
"#,
        expect![[r#"[Foo::bar#anchor](https://docs.rs/foo/*/foo/struct.Foo.html#anchor)"#]],
    );
    check_rewrite(
        r#"
//- /main.rs crate:foo
/// [method](Foo::bar)
pub struct $0Foo;

impl Foo {
    fn bar() {}
}
"#,
        expect![[r#"[method](https://docs.rs/foo/*/foo/struct.Foo.html#method.bar)"#]],
    );
}
