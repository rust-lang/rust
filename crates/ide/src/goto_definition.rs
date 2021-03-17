use either::Either;
use hir::{HasAttrs, ModuleDef, Semantics};
use ide_db::{
    defs::{Definition, NameClass, NameRefClass},
    RootDatabase,
};
use syntax::{
    ast, match_ast, AstNode, AstToken, SyntaxKind::*, SyntaxToken, TextSize, TokenAtOffset, T,
};

use crate::{
    display::TryToNav, doc_links::extract_definitions_from_markdown, runnables::doc_owner_to_def,
    FilePosition, NavigationTarget, RangeInfo,
};

// Feature: Go to Definition
//
// Navigates to the definition of an identifier.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[F12]
// |===
pub(crate) fn goto_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let original_token = pick_best(file.token_at_offset(position.offset))?;
    let token = sema.descend_into_macros(original_token.clone());
    let parent = token.parent()?;
    if let Some(comment) = ast::Comment::cast(token) {
        let nav = def_for_doc_comment(&sema, position, &comment)?.try_to_nav(db)?;
        return Some(RangeInfo::new(original_token.text_range(), vec![nav]));
    }

    let nav = match_ast! {
        match parent {
            ast::NameRef(name_ref) => {
                reference_definition(&sema, Either::Right(&name_ref))
            },
            ast::Name(name) => {
                let def = NameClass::classify(&sema, &name)?.referenced_or_defined(sema.db);
                def.try_to_nav(sema.db)
            },
            ast::Lifetime(lt) => if let Some(name_class) = NameClass::classify_lifetime(&sema, &lt) {
                let def = name_class.referenced_or_defined(sema.db);
                def.try_to_nav(sema.db)
            } else {
                reference_definition(&sema, Either::Left(&lt))
            },
            _ => return None,
        }
    };

    Some(RangeInfo::new(original_token.text_range(), nav.into_iter().collect()))
}

fn def_for_doc_comment(
    sema: &Semantics<RootDatabase>,
    position: FilePosition,
    doc_comment: &ast::Comment,
) -> Option<hir::ModuleDef> {
    let parent = doc_comment.syntax().parent()?;
    let (link, ns) = extract_positioned_link_from_comment(position, doc_comment)?;

    let def = doc_owner_to_def(sema, parent)?;
    match def {
        Definition::ModuleDef(def) => match def {
            ModuleDef::Module(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::Function(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::Adt(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::Variant(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::Const(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::Static(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::Trait(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::TypeAlias(it) => it.resolve_doc_path(sema.db, &link, ns),
            ModuleDef::BuiltinType(_) => return None,
        },
        Definition::Macro(it) => it.resolve_doc_path(sema.db, &link, ns),
        Definition::Field(it) => it.resolve_doc_path(sema.db, &link, ns),
        Definition::SelfType(_)
        | Definition::Local(_)
        | Definition::GenericParam(_)
        | Definition::Label(_) => return None,
    }
}

fn extract_positioned_link_from_comment(
    position: FilePosition,
    comment: &ast::Comment,
) -> Option<(String, Option<hir::Namespace>)> {
    let comment_range = comment.syntax().text_range();
    let doc_comment = comment.doc_comment()?;
    let def_links = extract_definitions_from_markdown(doc_comment);
    let start = comment_range.start() + TextSize::from(comment.prefix().len() as u32);
    let (def_link, ns, _) = def_links.iter().min_by_key(|(_, _, def_link_range)| {
        let matched_position = start + TextSize::from(def_link_range.start as u32);
        position.offset.checked_sub(matched_position).unwrap_or_else(|| comment_range.end())
    })?;
    Some((def_link.to_string(), *ns))
}

fn pick_best(tokens: TokenAtOffset<SyntaxToken>) -> Option<SyntaxToken> {
    return tokens.max_by_key(priority);
    fn priority(n: &SyntaxToken) -> usize {
        match n.kind() {
            IDENT | INT_NUMBER | LIFETIME_IDENT | T![self] | COMMENT => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        }
    }
}

pub(crate) fn reference_definition(
    sema: &Semantics<RootDatabase>,
    name_ref: Either<&ast::Lifetime, &ast::NameRef>,
) -> Option<NavigationTarget> {
    let name_kind = name_ref.either(
        |lifetime| NameRefClass::classify_lifetime(sema, lifetime),
        |name_ref| NameRefClass::classify(sema, name_ref),
    )?;
    let def = name_kind.referenced(sema.db);
    def.try_to_nav(sema.db)
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;

    use crate::fixture;

    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::nav_target_annotation(ra_fixture);
        let mut navs =
            analysis.goto_definition(position).unwrap().expect("no definition found").info;
        if navs.len() == 0 {
            panic!("unresolved reference")
        }
        assert_eq!(navs.len(), 1);

        let nav = navs.pop().unwrap();
        assert_eq!(expected, FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() });
    }

    #[test]
    fn goto_def_for_extern_crate() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std$0;
//- /std/lib.rs crate:std
// empty
//^ file
"#,
        )
    }

    #[test]
    fn goto_def_for_renamed_extern_crate() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std as abc$0;
//- /std/lib.rs crate:std
// empty
//^ file
"#,
        )
    }

    #[test]
    fn goto_def_in_items() {
        check(
            r#"
struct Foo;
     //^^^
enum E { X(Foo$0) }
"#,
        );
    }

    #[test]
    fn goto_def_at_start_of_item() {
        check(
            r#"
struct Foo;
     //^^^
enum E { X($0Foo) }
"#,
        );
    }

    #[test]
    fn goto_definition_resolves_correct_name() {
        check(
            r#"
//- /lib.rs
use a::Foo;
mod a;
mod b;
enum E { X(Foo$0) }

//- /a.rs
struct Foo;
     //^^^
//- /b.rs
struct Foo;
"#,
        );
    }

    #[test]
    fn goto_def_for_module_declaration() {
        check(
            r#"
//- /lib.rs
mod $0foo;

//- /foo.rs
// empty
//^ file
"#,
        );

        check(
            r#"
//- /lib.rs
mod $0foo;

//- /foo/mod.rs
// empty
//^ file
"#,
        );
    }

    #[test]
    fn goto_def_for_macros() {
        check(
            r#"
macro_rules! foo { () => { () } }
           //^^^
fn bar() {
    $0foo!();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_macros_from_other_crates() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo::foo;
fn bar() {
    $0foo!();
}

//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! foo { () => { () } }
           //^^^
"#,
        );
    }

    #[test]
    fn goto_def_for_macros_in_use_tree() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo::foo$0;

//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! foo { () => { () } }
           //^^^
"#,
        );
    }

    #[test]
    fn goto_def_for_macro_defined_fn_with_arg() {
        check(
            r#"
//- /lib.rs
macro_rules! define_fn {
    ($name:ident) => (fn $name() {})
}

define_fn!(foo);
         //^^^

fn bar() {
   $0foo();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_macro_defined_fn_no_arg() {
        check(
            r#"
//- /lib.rs
macro_rules! define_fn {
    () => (fn foo() {})
}

  define_fn!();
//^^^^^^^^^^^^^

fn bar() {
   $0foo();
}
"#,
        );
    }

    #[test]
    fn goto_definition_works_for_macro_inside_pattern() {
        check(
            r#"
//- /lib.rs
macro_rules! foo {() => {0}}
           //^^^

fn bar() {
    match (0,1) {
        ($0foo!(), _) => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_definition_works_for_macro_inside_match_arm_lhs() {
        check(
            r#"
//- /lib.rs
macro_rules! foo {() => {0}}
           //^^^
fn bar() {
    match 0 {
        $0foo!() => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_use_alias() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo as bar$0;

//- /foo/lib.rs crate:foo
// empty
//^ file
"#,
        );
    }

    #[test]
    fn goto_def_for_use_alias_foo_macro() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
use foo::foo as bar$0;

//- /foo/lib.rs crate:foo
#[macro_export]
macro_rules! foo { () => { () } }
           //^^^
"#,
        );
    }

    #[test]
    fn goto_def_for_methods() {
        check(
            r#"
struct Foo;
impl Foo {
    fn frobnicate(&self) { }
     //^^^^^^^^^^
}

fn bar(foo: &Foo) {
    foo.frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_fields() {
        check(
            r#"
struct Foo {
    spam: u32,
} //^^^^

fn bar(foo: &Foo) {
    foo.spam$0;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_record_fields() {
        check(
            r#"
//- /lib.rs
struct Foo {
    spam: u32,
} //^^^^

fn bar() -> Foo {
    Foo {
        spam$0: 0,
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_record_pat_fields() {
        check(
            r#"
//- /lib.rs
struct Foo {
    spam: u32,
} //^^^^

fn bar(foo: Foo) -> Foo {
    let Foo { spam$0: _, } = foo
}
"#,
        );
    }

    #[test]
    fn goto_def_for_record_fields_macros() {
        check(
            r"
macro_rules! m { () => { 92 };}
struct Foo { spam: u32 }
           //^^^^

fn bar() -> Foo {
    Foo { spam$0: m!() }
}
",
        );
    }

    #[test]
    fn goto_for_tuple_fields() {
        check(
            r#"
struct Foo(u32);
         //^^^

fn bar() {
    let foo = Foo(0);
    foo.$00;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_ufcs_inherent_methods() {
        check(
            r#"
struct Foo;
impl Foo {
    fn frobnicate() { }
}    //^^^^^^^^^^

fn bar(foo: &Foo) {
    Foo::frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_ufcs_trait_methods_through_traits() {
        check(
            r#"
trait Foo {
    fn frobnicate();
}    //^^^^^^^^^^

fn bar() {
    Foo::frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_ufcs_trait_methods_through_self() {
        check(
            r#"
struct Foo;
trait Trait {
    fn frobnicate();
}    //^^^^^^^^^^
impl Trait for Foo {}

fn bar() {
    Foo::frobnicate$0();
}
"#,
        );
    }

    #[test]
    fn goto_definition_on_self() {
        check(
            r#"
struct Foo;
impl Foo {
   //^^^
    pub fn new() -> Self {
        Self$0 {}
    }
}
"#,
        );
        check(
            r#"
struct Foo;
impl Foo {
   //^^^
    pub fn new() -> Self$0 {
        Self {}
    }
}
"#,
        );

        check(
            r#"
enum Foo { A }
impl Foo {
   //^^^
    pub fn new() -> Self$0 {
        Foo::A
    }
}
"#,
        );

        check(
            r#"
enum Foo { A }
impl Foo {
   //^^^
    pub fn thing(a: &Self$0) {
    }
}
"#,
        );
    }

    #[test]
    fn goto_definition_on_self_in_trait_impl() {
        check(
            r#"
struct Foo;
trait Make {
    fn new() -> Self;
}
impl Make for Foo {
            //^^^
    fn new() -> Self {
        Self$0 {}
    }
}
"#,
        );

        check(
            r#"
struct Foo;
trait Make {
    fn new() -> Self;
}
impl Make for Foo {
            //^^^
    fn new() -> Self$0 {
        Self {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_when_used_on_definition_name_itself() {
        check(
            r#"
struct Foo$0 { value: u32 }
     //^^^
            "#,
        );

        check(
            r#"
struct Foo {
    field$0: string,
} //^^^^^
"#,
        );

        check(
            r#"
fn foo_test$0() { }
 //^^^^^^^^
"#,
        );

        check(
            r#"
enum Foo$0 { Variant }
   //^^^
"#,
        );

        check(
            r#"
enum Foo {
    Variant1,
    Variant2$0,
  //^^^^^^^^
    Variant3,
}
"#,
        );

        check(
            r#"
static INNER$0: &str = "";
     //^^^^^
"#,
        );

        check(
            r#"
const INNER$0: &str = "";
    //^^^^^
"#,
        );

        check(
            r#"
type Thing$0 = Option<()>;
   //^^^^^
"#,
        );

        check(
            r#"
trait Foo$0 { }
    //^^^
"#,
        );

        check(
            r#"
mod bar$0 { }
  //^^^
"#,
        );
    }

    #[test]
    fn goto_from_macro() {
        check(
            r#"
macro_rules! id {
    ($($tt:tt)*) => { $($tt)* }
}
fn foo() {}
 //^^^
id! {
    fn bar() {
        fo$0o();
    }
}
mod confuse_index { fn foo(); }
"#,
        );
    }

    #[test]
    fn goto_through_format() {
        check(
            r#"
#[macro_export]
macro_rules! format {
    ($($arg:tt)*) => ($crate::fmt::format($crate::__export::format_args!($($arg)*)))
}
#[rustc_builtin_macro]
#[macro_export]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}
pub mod __export {
    pub use crate::format_args;
    fn foo() {} // for index confusion
}
fn foo() -> i8 {}
 //^^^
fn test() {
    format!("{}", fo$0o())
}
"#,
        );
    }

    #[test]
    fn goto_through_included_file() {
        check(
            r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {}

  include!("foo.rs");
//^^^^^^^^^^^^^^^^^^^

fn f() {
    foo$0();
}

mod confuse_index {
    pub fn foo() {}
}

//- /foo.rs
fn foo() {}
        "#,
        );
    }

    #[test]
    fn goto_for_type_param() {
        check(
            r#"
struct Foo<T: Clone> { t: $0T }
         //^
"#,
        );
    }

    #[test]
    fn goto_within_macro() {
        check(
            r#"
macro_rules! id {
    ($($tt:tt)*) => ($($tt)*)
}

fn foo() {
    let x = 1;
      //^
    id!({
        let y = $0x;
        let z = y;
    });
}
"#,
        );

        check(
            r#"
macro_rules! id {
    ($($tt:tt)*) => ($($tt)*)
}

fn foo() {
    let x = 1;
    id!({
        let y = x;
          //^
        let z = $0y;
    });
}
"#,
        );
    }

    #[test]
    fn goto_def_in_local_fn() {
        check(
            r#"
fn main() {
    fn foo() {
        let x = 92;
          //^
        $0x;
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_in_local_macro() {
        check(
            r#"
fn bar() {
    macro_rules! foo { () => { () } }
               //^^^
    $0foo!();
}
"#,
        );
    }

    #[test]
    fn goto_def_for_field_init_shorthand() {
        check(
            r#"
struct Foo { x: i32 }
fn main() {
    let x = 92;
      //^
    Foo { x$0 };
}
"#,
        )
    }

    #[test]
    fn goto_def_for_enum_variant_field() {
        check(
            r#"
enum Foo {
    Bar { x: i32 }
}       //^
fn baz(foo: Foo) {
    match foo {
        Foo::Bar { x$0 } => x
    };
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_pattern_const() {
        check(
            r#"
enum Foo { Bar }
         //^^^
impl Foo {
    fn baz(self) {
        match self { Self::Bar$0 => {} }
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_pattern_record() {
        check(
            r#"
enum Foo { Bar { val: i32 } }
         //^^^
impl Foo {
    fn baz(self) -> i32 {
        match self { Self::Bar$0 { val } => {} }
    }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_expr_const() {
        check(
            r#"
enum Foo { Bar }
         //^^^
impl Foo {
    fn baz(self) { Self::Bar$0; }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_enum_variant_self_expr_record() {
        check(
            r#"
enum Foo { Bar { val: i32 } }
         //^^^
impl Foo {
    fn baz(self) { Self::Bar$0 {val: 4}; }
}
"#,
        );
    }

    #[test]
    fn goto_def_for_type_alias_generic_parameter() {
        check(
            r#"
type Alias<T> = T$0;
         //^
"#,
        )
    }

    #[test]
    fn goto_def_for_macro_container() {
        check(
            r#"
//- /lib.rs crate:main deps:foo
foo::module$0::mac!();

//- /foo/lib.rs crate:foo
pub mod module {
      //^^^^^^
    #[macro_export]
    macro_rules! _mac { () => { () } }
    pub use crate::_mac as mac;
}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_in_path() {
        check(
            r#"
trait Iterator {
    type Item;
       //^^^^
}

fn f() -> impl Iterator<Item$0 = u8> {}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_in_path_multiple() {
        check(
            r#"
trait Iterator {
    type A;
       //^
    type B;
}

fn f() -> impl Iterator<A$0 = u8, B = ()> {}
"#,
        );
        check(
            r#"
trait Iterator {
    type A;
    type B;
       //^
}

fn f() -> impl Iterator<A = u8, B$0 = ()> {}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_ufcs() {
        check(
            r#"
trait Iterator {
    type Item;
       //^^^^
}

fn g() -> <() as Iterator<Item$0 = ()>>::Item {}
"#,
        );
    }

    #[test]
    fn goto_def_for_assoc_ty_ufcs_multiple() {
        check(
            r#"
trait Iterator {
    type A;
       //^
    type B;
}

fn g() -> <() as Iterator<A$0 = (), B = u8>>::B {}
"#,
        );
        check(
            r#"
trait Iterator {
    type A;
    type B;
       //^
}

fn g() -> <() as Iterator<A = (), B$0 = u8>>::A {}
"#,
        );
    }

    #[test]
    fn goto_self_param_ty_specified() {
        check(
            r#"
struct Foo {}

impl Foo {
    fn bar(self: &Foo) {
         //^^^^
        let foo = sel$0f;
    }
}"#,
        )
    }

    #[test]
    fn goto_self_param_on_decl() {
        check(
            r#"
struct Foo {}

impl Foo {
    fn bar(&self$0) {
          //^^^^
    }
}"#,
        )
    }

    #[test]
    fn goto_lifetime_param_on_decl() {
        check(
            r#"
fn foo<'foobar$0>(_: &'foobar ()) {
     //^^^^^^^
}"#,
        )
    }

    #[test]
    fn goto_lifetime_param_decl() {
        check(
            r#"
fn foo<'foobar>(_: &'foobar$0 ()) {
     //^^^^^^^
}"#,
        )
    }

    #[test]
    fn goto_lifetime_param_decl_nested() {
        check(
            r#"
fn foo<'foobar>(_: &'foobar ()) {
    fn foo<'foobar>(_: &'foobar$0 ()) {}
         //^^^^^^^
}"#,
        )
    }

    #[test]
    #[ignore] // requires the HIR to somehow track these hrtb lifetimes
    fn goto_lifetime_hrtb() {
        check(
            r#"trait Foo<T> {}
fn foo<T>() where for<'a> T: Foo<&'a$0 (u8, u16)>, {}
                    //^^
"#,
        );
        check(
            r#"trait Foo<T> {}
fn foo<T>() where for<'a$0> T: Foo<&'a (u8, u16)>, {}
                    //^^
"#,
        );
    }

    #[test]
    #[ignore] // requires ForTypes to be implemented
    fn goto_lifetime_hrtb_for_type() {
        check(
            r#"trait Foo<T> {}
fn foo<T>() where T: for<'a> Foo<&'a$0 (u8, u16)>, {}
                       //^^
"#,
        );
    }

    #[test]
    fn goto_label() {
        check(
            r#"
fn foo<'foo>(_: &'foo ()) {
    'foo: {
  //^^^^
        'bar: loop {
            break 'foo$0;
        }
    }
}"#,
        )
    }

    #[test]
    fn goto_def_for_intra_doc_link_same_file() {
        check(
            r#"
/// Blah, [`bar`](bar) .. [`foo`](foo)$0 has [`bar`](bar)
pub fn bar() { }

/// You might want to see [`std::fs::read()`] too.
pub fn foo() { }
     //^^^

}"#,
        )
    }

    #[test]
    fn goto_def_for_intra_doc_link_inner() {
        check(
            r#"
//- /main.rs
mod m;
struct S;
     //^

//- /m.rs
//! [`super::S$0`]
"#,
        )
    }

    #[test]
    fn goto_incomplete_field() {
        check(
            r#"
struct A { a: u32 }
         //^
fn foo() { A { a$0: }; }
"#,
        )
    }
}
