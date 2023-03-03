use std::mem::discriminant;

use crate::{doc_links::token_as_doc_comment, FilePosition, NavigationTarget, RangeInfo, TryToNav};
use hir::{AsAssocItem, AssocItem, Semantics};
use ide_db::{
    base_db::{AnchoredPath, FileId, FileLoader},
    defs::{Definition, IdentClass},
    helpers::pick_best_token,
    RootDatabase,
};
use itertools::Itertools;
use syntax::{ast, AstNode, AstToken, SyntaxKind::*, SyntaxToken, TextRange, T};

// Feature: Go to Definition
//
// Navigates to the definition of an identifier.
//
// For outline modules, this will navigate to the source file of the module.
//
// |===
// | Editor  | Shortcut
//
// | VS Code | kbd:[F12]
// |===
//
// image::https://user-images.githubusercontent.com/48062697/113065563-025fbe00-91b1-11eb-83e4-a5a703610b23.gif[]
pub(crate) fn goto_definition(
    db: &RootDatabase,
    position: FilePosition,
) -> Option<RangeInfo<Vec<NavigationTarget>>> {
    let sema = &Semantics::new(db);
    let file = sema.parse(position.file_id).syntax().clone();
    let original_token =
        pick_best_token(file.token_at_offset(position.offset), |kind| match kind {
            IDENT
            | INT_NUMBER
            | LIFETIME_IDENT
            | T![self]
            | T![super]
            | T![crate]
            | T![Self]
            | COMMENT => 4,
            // index and prefix ops
            T!['['] | T![']'] | T![?] | T![*] | T![-] | T![!] => 3,
            kind if kind.is_keyword() => 2,
            T!['('] | T![')'] => 2,
            kind if kind.is_trivia() => 0,
            _ => 1,
        })?;
    if let Some(doc_comment) = token_as_doc_comment(&original_token) {
        return doc_comment.get_definition_with_descend_at(
            sema,
            position.offset,
            |def, _, link_range| {
                let nav = def.try_to_nav(db)?;
                Some(RangeInfo::new(link_range, vec![nav]))
            },
        );
    }
    let navs = sema
        .descend_into_macros(original_token.clone())
        .into_iter()
        .filter_map(|token| {
            let parent = token.parent()?;
            if let Some(tt) = ast::TokenTree::cast(parent) {
                if let Some(x) = try_lookup_include_path(sema, tt, token.clone(), position.file_id)
                {
                    return Some(vec![x]);
                }
            }
            Some(
                IdentClass::classify_token(sema, &token)?
                    .definitions()
                    .into_iter()
                    .flat_map(|def| {
                        try_filter_trait_item_definition(sema, &def)
                            .unwrap_or_else(|| def_to_nav(sema.db, def))
                    })
                    .collect(),
            )
        })
        .flatten()
        .unique()
        .collect::<Vec<NavigationTarget>>();

    Some(RangeInfo::new(original_token.text_range(), navs))
}

fn try_lookup_include_path(
    sema: &Semantics<'_, RootDatabase>,
    tt: ast::TokenTree,
    token: SyntaxToken,
    file_id: FileId,
) -> Option<NavigationTarget> {
    let token = ast::String::cast(token)?;
    let path = token.value()?.into_owned();
    let macro_call = tt.syntax().parent().and_then(ast::MacroCall::cast)?;
    let name = macro_call.path()?.segment()?.name_ref()?;
    if !matches!(&*name.text(), "include" | "include_str" | "include_bytes") {
        return None;
    }

    // Ignore non-built-in macros to account for shadowing
    if let Some(it) = sema.resolve_macro_call(&macro_call) {
        if !matches!(it.kind(sema.db), hir::MacroKind::BuiltIn) {
            return None;
        }
    }

    let file_id = sema.db.resolve_path(AnchoredPath { anchor: file_id, path: &path })?;
    let size = sema.db.file_text(file_id).len().try_into().ok()?;
    Some(NavigationTarget {
        file_id,
        full_range: TextRange::new(0.into(), size),
        name: path.into(),
        focus_range: None,
        kind: None,
        container_name: None,
        description: None,
        docs: None,
    })
}
/// finds the trait definition of an impl'd item, except function
/// e.g.
/// ```rust
/// trait A { type a; }
/// struct S;
/// impl A for S { type a = i32; } // <-- on this associate type, will get the location of a in the trait
/// ```
fn try_filter_trait_item_definition(
    sema: &Semantics<'_, RootDatabase>,
    def: &Definition,
) -> Option<Vec<NavigationTarget>> {
    let db = sema.db;
    let assoc = def.as_assoc_item(db)?;
    match assoc {
        AssocItem::Function(..) => None,
        AssocItem::Const(..) | AssocItem::TypeAlias(..) => {
            let imp = match assoc.container(db) {
                hir::AssocItemContainer::Impl(imp) => imp,
                _ => return None,
            };
            let trait_ = imp.trait_(db)?;
            let name = def.name(db)?;
            let discri_value = discriminant(&assoc);
            trait_
                .items(db)
                .iter()
                .filter(|itm| discriminant(*itm) == discri_value)
                .find_map(|itm| (itm.name(db)? == name).then(|| itm.try_to_nav(db)).flatten())
                .map(|it| vec![it])
        }
    }
}

fn def_to_nav(db: &RootDatabase, def: Definition) -> Vec<NavigationTarget> {
    def.try_to_nav(db).map(|it| vec![it]).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::FileRange;
    use itertools::Itertools;

    use crate::fixture;

    #[track_caller]
    fn check(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let navs = analysis.goto_definition(position).unwrap().expect("no definition found").info;

        let cmp = |&FileRange { file_id, range }: &_| (file_id, range.start());
        let navs = navs
            .into_iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        let expected = expected
            .into_iter()
            .map(|(FileRange { file_id, range }, _)| FileRange { file_id, range })
            .sorted_by_key(cmp)
            .collect::<Vec<_>>();
        assert_eq!(expected, navs);
    }

    fn check_unresolved(ra_fixture: &str) {
        let (analysis, position) = fixture::position(ra_fixture);
        let navs = analysis.goto_definition(position).unwrap().expect("no definition found").info;

        assert!(navs.is_empty(), "didn't expect this to resolve anywhere: {navs:?}")
    }

    #[test]
    fn goto_def_if_items_same_name() {
        check(
            r#"
trait Trait {
    type A;
    const A: i32;
        //^
}

struct T;
impl Trait for T {
    type A = i32;
    const A$0: i32 = -9;
}"#,
        );
    }
    #[test]
    fn goto_def_in_mac_call_in_attr_invoc() {
        check(
            r#"
//- proc_macros: identity
pub struct Struct {
        // ^^^^^^
    field: i32,
}

macro_rules! identity {
    ($($tt:tt)*) => {$($tt)*};
}

#[proc_macros::identity]
fn function() {
    identity!(Struct$0 { field: 0 });
}

"#,
        )
    }

    #[test]
    fn goto_def_for_extern_crate() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std$0;
//- /std/lib.rs crate:std
// empty
//^file
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
//^file
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
pub struct Foo;
         //^^^
//- /b.rs
pub struct Foo;
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
//^file
"#,
        );

        check(
            r#"
//- /lib.rs
mod $0foo;

//- /foo/mod.rs
// empty
//^file
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
//^file
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
trait Foo$0 = ;
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
           //^
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
        //^
}
fn baz(foo: Foo) {
    match foo {
        Foo::Bar { x$0 } => x
                 //^
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
    fn goto_def_for_super_assoc_ty_in_path() {
        check(
            r#"
trait Super {
    type Item;
       //^^^^
}

trait Sub: Super {}

fn f() -> impl Sub<Item$0 = u8> {}
"#,
        );
    }

    #[test]
    fn goto_def_for_module_declaration_in_path_if_types_and_values_same_name() {
        check(
            r#"
mod bar {
    pub struct Foo {}
             //^^^
    pub fn Foo() {}
}

fn baz() {
    let _foo_enum: bar::Foo$0 = bar::Foo {};
}
        "#,
        )
    }

    #[test]
    fn unknown_assoc_ty() {
        check_unresolved(
            r#"
trait Iterator { type Item; }
fn f() -> impl Iterator<Invalid$0 = u8> {}
"#,
        )
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
    fn goto_lifetime_hrtb() {
        // FIXME: requires the HIR to somehow track these hrtb lifetimes
        check_unresolved(
            r#"
trait Foo<T> {}
fn foo<T>() where for<'a> T: Foo<&'a$0 (u8, u16)>, {}
                    //^^
"#,
        );
        check_unresolved(
            r#"
trait Foo<T> {}
fn foo<T>() where for<'a$0> T: Foo<&'a (u8, u16)>, {}
                    //^^
"#,
        );
    }

    #[test]
    fn goto_lifetime_hrtb_for_type() {
        // FIXME: requires ForTypes to be implemented
        check_unresolved(
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
/// Blah, [`bar`](bar) .. [`foo`](foo$0) has [`bar`](bar)
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

    #[test]
    fn goto_proc_macro() {
        check(
            r#"
//- /main.rs crate:main deps:mac
use mac::fn_macro;

fn_macro$0!();

//- /mac.rs crate:mac
#![crate_type="proc-macro"]
#[proc_macro]
fn fn_macro() {}
 //^^^^^^^^
            "#,
        )
    }

    #[test]
    fn goto_intra_doc_links() {
        check(
            r#"

pub mod theitem {
    /// This is the item. Cool!
    pub struct TheItem;
             //^^^^^^^
}

/// Gives you a [`TheItem$0`].
///
/// [`TheItem`]: theitem::TheItem
pub fn gimme() -> theitem::TheItem {
    theitem::TheItem
}
"#,
        );
    }

    #[test]
    fn goto_ident_from_pat_macro() {
        check(
            r#"
macro_rules! pat {
    ($name:ident) => { Enum::Variant1($name) }
}

enum Enum {
    Variant1(u8),
    Variant2,
}

fn f(e: Enum) {
    match e {
        pat!(bind) => {
           //^^^^
            bind$0
        }
        Enum::Variant2 => {}
    }
}
"#,
        );
    }

    #[test]
    fn goto_include() {
        check(
            r#"
//- /main.rs

#[rustc_builtin_macro]
macro_rules! include_str {}

fn main() {
    let str = include_str!("foo.txt$0");
}
//- /foo.txt
// empty
//^file
"#,
        );
    }

    #[test]
    fn goto_doc_include_str() {
        check(
            r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include_str {}

#[doc = include_str!("docs.md$0")]
struct Item;

//- /docs.md
// docs
//^file
"#,
        );
    }

    #[test]
    fn goto_shadow_include() {
        check(
            r#"
//- /main.rs
macro_rules! include {
    ("included.rs") => {}
}

include!("included.rs$0");

//- /included.rs
// empty
"#,
        );
    }

    mod goto_impl_of_trait_fn {
        use super::check;
        #[test]
        fn cursor_on_impl() {
            check(
                r#"
trait Twait {
    fn a();
}

struct Stwuct;

impl Twait for Stwuct {
    fn a$0();
     //^
}
        "#,
            );
        }
        #[test]
        fn method_call() {
            check(
                r#"
trait Twait {
    fn a(&self);
}

struct Stwuct;

impl Twait for Stwuct {
    fn a(&self){};
     //^
}
fn f() {
    let s = Stwuct;
    s.a$0();
}
        "#,
            );
        }
        #[test]
        fn path_call() {
            check(
                r#"
trait Twait {
    fn a(&self);
}

struct Stwuct;

impl Twait for Stwuct {
    fn a(&self){};
     //^
}
fn f() {
    let s = Stwuct;
    Stwuct::a$0(&s);
}
        "#,
            );
        }
        #[test]
        fn where_clause_can_work() {
            check(
                r#"
trait G {
    fn g(&self);
}
trait Bound{}
trait EA{}
struct Gen<T>(T);
impl <T:EA> G for Gen<T> {
    fn g(&self) {
    }
}
impl <T> G for Gen<T>
where T : Bound
{
    fn g(&self){
     //^
    }
}
struct A;
impl Bound for A{}
fn f() {
    let gen = Gen::<A>(A);
    gen.g$0();
}
                "#,
            );
        }
        #[test]
        fn wc_case_is_ok() {
            check(
                r#"
trait G {
    fn g(&self);
}
trait BParent{}
trait Bound: BParent{}
struct Gen<T>(T);
impl <T> G for Gen<T>
where T : Bound
{
    fn g(&self){
     //^
    }
}
struct A;
impl Bound for A{}
fn f() {
    let gen = Gen::<A>(A);
    gen.g$0();
}
"#,
            );
        }

        #[test]
        fn method_call_defaulted() {
            check(
                r#"
trait Twait {
    fn a(&self) {}
     //^
}

struct Stwuct;

impl Twait for Stwuct {
}
fn f() {
    let s = Stwuct;
    s.a$0();
}
        "#,
            );
        }

        #[test]
        fn method_call_on_generic() {
            check(
                r#"
trait Twait {
    fn a(&self) {}
     //^
}

fn f<T: Twait>(s: T) {
    s.a$0();
}
        "#,
            );
        }
    }

    #[test]
    fn goto_def_of_trait_impl_const() {
        check(
            r#"
trait Twait {
    const NOMS: bool;
       // ^^^^
}

struct Stwuct;

impl Twait for Stwuct {
    const NOMS$0: bool = true;
}
"#,
        );
    }

    #[test]
    fn goto_def_of_trait_impl_type_alias() {
        check(
            r#"
trait Twait {
    type IsBad;
      // ^^^^^
}

struct Stwuct;

impl Twait for Stwuct {
    type IsBad$0 = !;
}
"#,
        );
    }

    #[test]
    fn goto_def_derive_input() {
        check(
            r#"
        //- minicore:derive
        #[rustc_builtin_macro]
        pub macro Copy {}
               // ^^^^
        #[derive(Copy$0)]
        struct Foo;
                    "#,
        );
        check(
            r#"
//- minicore:derive
#[rustc_builtin_macro]
pub macro Copy {}
       // ^^^^
#[cfg_attr(feature = "false", derive)]
#[derive(Copy$0)]
struct Foo;
            "#,
        );
        check(
            r#"
//- minicore:derive
mod foo {
    #[rustc_builtin_macro]
    pub macro Copy {}
           // ^^^^
}
#[derive(foo::Copy$0)]
struct Foo;
            "#,
        );
        check(
            r#"
//- minicore:derive
mod foo {
 // ^^^
    #[rustc_builtin_macro]
    pub macro Copy {}
}
#[derive(foo$0::Copy)]
struct Foo;
            "#,
        );
    }

    #[test]
    fn goto_def_in_macro_multi() {
        check(
            r#"
struct Foo {
    foo: ()
  //^^^
}
macro_rules! foo {
    ($ident:ident) => {
        fn $ident(Foo { $ident }: Foo) {}
    }
}
foo!(foo$0);
   //^^^
   //^^^
"#,
        );
        check(
            r#"
fn bar() {}
 //^^^
struct bar;
     //^^^
macro_rules! foo {
    ($ident:ident) => {
        fn foo() {
            let _: $ident = $ident;
        }
    }
}

foo!(bar$0);
"#,
        );
    }

    #[test]
    fn goto_await_poll() {
        check(
            r#"
//- minicore: future

struct MyFut;

impl core::future::Future for MyFut {
    type Output = ();

    fn poll(
     //^^^^
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>
    ) -> std::task::Poll<Self::Output>
    {
        ()
    }
}

fn f() {
    MyFut.await$0;
}
"#,
        );
    }

    #[test]
    fn goto_await_into_future_poll() {
        check(
            r#"
//- minicore: future

struct Futurable;

impl core::future::IntoFuture for Futurable {
    type IntoFuture = MyFut;
}

struct MyFut;

impl core::future::Future for MyFut {
    type Output = ();

    fn poll(
     //^^^^
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>
    ) -> std::task::Poll<Self::Output>
    {
        ()
    }
}

fn f() {
    Futurable.await$0;
}
"#,
        );
    }

    #[test]
    fn goto_try_op() {
        check(
            r#"
//- minicore: try

struct Struct;

impl core::ops::Try for Struct {
    fn branch(
     //^^^^^^
        self
    ) {}
}

fn f() {
    Struct?$0;
}
"#,
        );
    }

    #[test]
    fn goto_index_op() {
        check(
            r#"
//- minicore: index

struct Struct;

impl core::ops::Index<usize> for Struct {
    fn index(
     //^^^^^
        self
    ) {}
}

fn f() {
    Struct[0]$0;
}
"#,
        );
    }

    #[test]
    fn goto_prefix_op() {
        check(
            r#"
//- minicore: deref

struct Struct;

impl core::ops::Deref for Struct {
    fn deref(
     //^^^^^
        self
    ) {}
}

fn f() {
    $0*Struct;
}
"#,
        );
    }

    #[test]
    fn goto_bin_op() {
        check(
            r#"
//- minicore: add

struct Struct;

impl core::ops::Add for Struct {
    fn add(
     //^^^
        self
    ) {}
}

fn f() {
    Struct +$0 Struct;
}
"#,
        );
    }

    #[test]
    fn goto_bin_op_multiple_impl() {
        check(
            r#"
//- minicore: add
struct S;
impl core::ops::Add for S {
    fn add(
     //^^^
    ) {}
}
impl core::ops::Add<usize> for S {
    fn add(
    ) {}
}

fn f() {
    S +$0 S
}
"#,
        );

        check(
            r#"
//- minicore: add
struct S;
impl core::ops::Add for S {
    fn add(
    ) {}
}
impl core::ops::Add<usize> for S {
    fn add(
     //^^^
    ) {}
}

fn f() {
    S +$0 0usize
}
"#,
        );
    }

    #[test]
    fn path_call_multiple_trait_impl() {
        check(
            r#"
trait Trait<T> {
    fn f(_: T);
}
impl Trait<i32> for usize {
    fn f(_: i32) {}
     //^
}
impl Trait<i64> for usize {
    fn f(_: i64) {}
}
fn main() {
    usize::f$0(0i32);
}
"#,
        );

        check(
            r#"
trait Trait<T> {
    fn f(_: T);
}
impl Trait<i32> for usize {
    fn f(_: i32) {}
}
impl Trait<i64> for usize {
    fn f(_: i64) {}
     //^
}
fn main() {
    usize::f$0(0i64);
}
"#,
        )
    }

    #[test]
    fn query_impls_in_nearest_block() {
        check(
            r#"
struct S1;
impl S1 {
    fn e() -> () {}
}
fn f1() {
    struct S1;
    impl S1 {
        fn e() -> () {}
         //^
    }
    fn f2() {
        fn f3() {
            S1::e$0();
        }
    }
}
"#,
        );

        check(
            r#"
struct S1;
impl S1 {
    fn e() -> () {}
}
fn f1() {
    struct S1;
    impl S1 {
        fn e() -> () {}
         //^
    }
    fn f2() {
        struct S2;
        S1::e$0();
    }
}
fn f12() {
    struct S1;
    impl S1 {
        fn e() -> () {}
    }
}
"#,
        );

        check(
            r#"
struct S1;
impl S1 {
    fn e() -> () {}
     //^
}
fn f2() {
    struct S2;
    S1::e$0();
}
"#,
        );
    }
}
