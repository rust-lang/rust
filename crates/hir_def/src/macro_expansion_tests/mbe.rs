//! Tests specific to declarative macros, aka macros by example. This covers
//! both stable `macro_rules!` macros as well as unstable `macro` macros.

mod tt_conversion;
mod matching;
mod meta_syntax;

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn mbe_smoke_test() {
    check(
        r#"
macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e { $e::$v(it) }
            }
        )*
    }
}
impl_froms!(TokenTree: Leaf, Subtree);
"#,
        expect![[r#"
macro_rules! impl_froms {
    ($e:ident: $($v:ident),*) => {
        $(
            impl From<$v> for $e {
                fn from(it: $v) -> $e { $e::$v(it) }
            }
        )*
    }
}
impl From<Leaf> for TokenTree {
    fn from(it:Leaf) -> TokenTree {
        TokenTree::Leaf(it)
    }
}
impl From<Subtree> for TokenTree {
    fn from(it:Subtree) -> TokenTree {
        TokenTree::Subtree(it)
    }
}
"#]],
    );
}

#[test]
fn expansion_does_not_parse_as_expression() {
    check(
        r#"
macro_rules! stmts {
    () => { let _ = 0; }
}

fn f() { let _ = stmts!(); }
"#,
        expect![[r#"
macro_rules! stmts {
    () => { let _ = 0; }
}

fn f() { let _ = /* error: could not convert tokens */; }
"#]],
    )
}

#[test]
fn wrong_nesting_level() {
    check(
        r#"
macro_rules! m {
    ($($i:ident);*) => ($i)
}
m!{a}
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident);*) => ($i)
}
/* error: expected simple binding, found nested binding `i` */
"#]],
    );
}

#[test]
fn match_by_first_token_literally() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (= $i:ident) => ( fn $i() {} );
    (+ $i:ident) => ( struct $i; )
}
m! { foo }
m! { = bar }
m! { + Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (= $i:ident) => ( fn $i() {} );
    (+ $i:ident) => ( struct $i; )
}
mod foo {}
fn bar() {}
struct Baz;
"#]],
    );
}

#[test]
fn match_by_last_token_literally() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    ($i:ident =) => ( fn $i() {} );
    ($i:ident +) => ( struct $i; )
}
m! { foo }
m! { bar = }
m! { Baz + }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    ($i:ident =) => ( fn $i() {} );
    ($i:ident +) => ( struct $i; )
}
mod foo {}
fn bar() {}
struct Baz;
"#]],
    );
}

#[test]
fn match_by_ident() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (spam $i:ident) => ( fn $i() {} );
    (eggs $i:ident) => ( struct $i; )
}
m! { foo }
m! { spam bar }
m! { eggs Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( mod $i {} );
    (spam $i:ident) => ( fn $i() {} );
    (eggs $i:ident) => ( struct $i; )
}
mod foo {}
fn bar() {}
struct Baz;
"#]],
    );
}

#[test]
fn match_by_separator_token() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ($(mod $i {} )*);
    ($($i:ident)#*) => ($(fn $i() {} )*);
    ($i:ident ,# $ j:ident) => ( struct $i; struct $ j; )
}

m! { foo, bar }

m! { foo# bar }

m! { Foo,# Bar }
"#,
        expect![[r##"
macro_rules! m {
    ($($i:ident),*) => ($(mod $i {} )*);
    ($($i:ident)#*) => ($(fn $i() {} )*);
    ($i:ident ,# $ j:ident) => ( struct $i; struct $ j; )
}

mod foo {}
mod bar {}

fn foo() {}
fn bar() {}

struct Foo;
struct Bar;
"##]],
    );
}

#[test]
fn test_match_group_pattern_with_multiple_defs() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( impl Bar { $(fn $i() {})* } );
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( impl Bar { $(fn $i() {})* } );
}
impl Bar {
    fn foo() {}
    fn bar() {}
}
"#]],
    );
}

#[test]
fn test_match_group_pattern_with_multiple_statement() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i ();)* } );
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i ();)* } );
}
fn baz() {
    foo();
    bar();
}
"#]],
    )
}

#[test]
fn test_match_group_pattern_with_multiple_statement_without_semi() {
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i() );* } );
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( fn baz() { $($i() );* } );
}
fn baz() {
    foo();
    bar()
}
"#]],
    )
}

#[test]
fn test_match_group_empty_fixed_token() {
    check(
        r#"
macro_rules! m {
    ($($i:ident)* #abc) => ( fn baz() { $($i ();)* } );
}
m!{#abc}
"#,
        expect![[r##"
macro_rules! m {
    ($($i:ident)* #abc) => ( fn baz() { $($i ();)* } );
}
fn baz() {}
"##]],
    )
}

#[test]
fn test_match_group_in_subtree() {
    check(
        r#"
macro_rules! m {
    (fn $name:ident { $($i:ident)* } ) => ( fn $name() { $($i ();)* } );
}
m! { fn baz { a b } }
"#,
        expect![[r#"
macro_rules! m {
    (fn $name:ident { $($i:ident)* } ) => ( fn $name() { $($i ();)* } );
}
fn baz() {
    a();
    b();
}
"#]],
    )
}

#[test]
fn test_expr_order() {
    check(
        r#"
macro_rules! m {
    ($ i:expr) => { fn bar() { $ i * 3; } }
}
// +tree
m! { 1 + 2 }
"#,
        expect![[r#"
macro_rules! m {
    ($ i:expr) => { fn bar() { $ i * 3; } }
}
fn bar() {
    1+2*3;
}
// MACRO_ITEMS@0..15
//   FN@0..15
//     FN_KW@0..2 "fn"
//     NAME@2..5
//       IDENT@2..5 "bar"
//     PARAM_LIST@5..7
//       L_PAREN@5..6 "("
//       R_PAREN@6..7 ")"
//     BLOCK_EXPR@7..15
//       STMT_LIST@7..15
//         L_CURLY@7..8 "{"
//         EXPR_STMT@8..14
//           BIN_EXPR@8..13
//             BIN_EXPR@8..11
//               LITERAL@8..9
//                 INT_NUMBER@8..9 "1"
//               PLUS@9..10 "+"
//               LITERAL@10..11
//                 INT_NUMBER@10..11 "2"
//             STAR@11..12 "*"
//             LITERAL@12..13
//               INT_NUMBER@12..13 "3"
//           SEMICOLON@13..14 ";"
//         R_CURLY@14..15 "}"

"#]],
    )
}

#[test]
fn test_match_group_with_multichar_sep() {
    check(
        r#"
macro_rules! m {
    (fn $name:ident { $($i:literal)* }) => ( fn $name() -> bool { $($i)&&* } );
}
m! (fn baz { true false } );
"#,
        expect![[r#"
macro_rules! m {
    (fn $name:ident { $($i:literal)* }) => ( fn $name() -> bool { $($i)&&* } );
}
fn baz() -> bool {
    true && false
}
"#]],
    );

    check(
        r#"
macro_rules! m {
    (fn $name:ident { $($i:literal)&&* }) => ( fn $name() -> bool { $($i)&&* } );
}
m! (fn baz { true && false } );
"#,
        expect![[r#"
macro_rules! m {
    (fn $name:ident { $($i:literal)&&* }) => ( fn $name() -> bool { $($i)&&* } );
}
fn baz() -> bool {
    true && false
}
"#]],
    );
}

#[test]
fn test_match_group_zero_match() {
    check(
        r#"
macro_rules! m { ( $($i:ident)* ) => (); }
m!();
"#,
        expect![[r#"
macro_rules! m { ( $($i:ident)* ) => (); }

"#]],
    );
}

#[test]
fn test_match_group_in_group() {
    check(
        r#"
macro_rules! m {
    [ $( ( $($i:ident)* ) )* ] => [ x![$( ( $($i)* ) )*]; ]
}
m! ( (a b) );
"#,
        expect![[r#"
macro_rules! m {
    [ $( ( $($i:ident)* ) )* ] => [ x![$( ( $($i)* ) )*]; ]
}
x![(a b)];
"#]],
    )
}

#[test]
fn test_expand_to_item_list() {
    check(
        r#"
macro_rules! structs {
    ($($i:ident),*) => { $(struct $i { field: u32 } )* }
}

// +tree
structs!(Foo, Bar);
            "#,
        expect![[r#"
macro_rules! structs {
    ($($i:ident),*) => { $(struct $i { field: u32 } )* }
}

struct Foo {
    field:u32
}
struct Bar {
    field:u32
}
// MACRO_ITEMS@0..40
//   STRUCT@0..20
//     STRUCT_KW@0..6 "struct"
//     NAME@6..9
//       IDENT@6..9 "Foo"
//     RECORD_FIELD_LIST@9..20
//       L_CURLY@9..10 "{"
//       RECORD_FIELD@10..19
//         NAME@10..15
//           IDENT@10..15 "field"
//         COLON@15..16 ":"
//         PATH_TYPE@16..19
//           PATH@16..19
//             PATH_SEGMENT@16..19
//               NAME_REF@16..19
//                 IDENT@16..19 "u32"
//       R_CURLY@19..20 "}"
//   STRUCT@20..40
//     STRUCT_KW@20..26 "struct"
//     NAME@26..29
//       IDENT@26..29 "Bar"
//     RECORD_FIELD_LIST@29..40
//       L_CURLY@29..30 "{"
//       RECORD_FIELD@30..39
//         NAME@30..35
//           IDENT@30..35 "field"
//         COLON@35..36 ":"
//         PATH_TYPE@36..39
//           PATH@36..39
//             PATH_SEGMENT@36..39
//               NAME_REF@36..39
//                 IDENT@36..39 "u32"
//       R_CURLY@39..40 "}"

            "#]],
    );
}

#[test]
fn test_two_idents() {
    check(
        r#"
macro_rules! m {
    ($i:ident, $j:ident) => { fn foo() { let a = $i; let b = $j; } }
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident, $j:ident) => { fn foo() { let a = $i; let b = $j; } }
}
fn foo() {
    let a = foo;
    let b = bar;
}
"#]],
    );
}

#[test]
fn test_tt_to_stmts() {
    check(
        r#"
macro_rules! m {
    () => {
        let a = 0;
        a = 10 + 1;
        a
    }
}

fn f() -> i32 {
    // +tree
    m!{}
}
"#,
        expect![[r#"
macro_rules! m {
    () => {
        let a = 0;
        a = 10 + 1;
        a
    }
}

fn f() -> i32 {
    let a = 0;
    a = 10+1;
    a
// MACRO_STMTS@0..15
//   LET_STMT@0..7
//     LET_KW@0..3 "let"
//     IDENT_PAT@3..4
//       NAME@3..4
//         IDENT@3..4 "a"
//     EQ@4..5 "="
//     LITERAL@5..6
//       INT_NUMBER@5..6 "0"
//     SEMICOLON@6..7 ";"
//   EXPR_STMT@7..14
//     BIN_EXPR@7..13
//       PATH_EXPR@7..8
//         PATH@7..8
//           PATH_SEGMENT@7..8
//             NAME_REF@7..8
//               IDENT@7..8 "a"
//       EQ@8..9 "="
//       BIN_EXPR@9..13
//         LITERAL@9..11
//           INT_NUMBER@9..11 "10"
//         PLUS@11..12 "+"
//         LITERAL@12..13
//           INT_NUMBER@12..13 "1"
//     SEMICOLON@13..14 ";"
//   PATH_EXPR@14..15
//     PATH@14..15
//       PATH_SEGMENT@14..15
//         NAME_REF@14..15
//           IDENT@14..15 "a"

}
"#]],
    );
}

#[test]
fn test_match_literal() {
    check(
        r#"
macro_rules! m {
    ('(') => { fn l_paren() {} }
}
m!['('];
"#,
        expect![[r#"
macro_rules! m {
    ('(') => { fn l_paren() {} }
}
fn l_paren() {}
"#]],
    );
}

#[test]
fn test_parse_macro_def_simple() {
    cov_mark::check!(parse_macro_def_simple);
    check(
        r#"
macro m($id:ident) { fn $id() {} }
m!(bar);
"#,
        expect![[r#"
macro m($id:ident) { fn $id() {} }
fn bar() {}
"#]],
    );
}

#[test]
fn test_parse_macro_def_rules() {
    cov_mark::check!(parse_macro_def_rules);

    check(
        r#"
macro m {
    ($id:ident) => { fn $id() {} }
}
m!(bar);
"#,
        expect![[r#"
macro m {
    ($id:ident) => { fn $id() {} }
}
fn bar() {}
"#]],
    );
}

#[test]
fn test_macro_2_0_panic_2015() {
    check(
        r#"
macro panic_2015 {
    () => (),
    (bar) => (),
}
panic_2015!(bar);
"#,
        expect![[r#"
macro panic_2015 {
    () => (),
    (bar) => (),
}

"#]],
    );
}

#[test]
fn test_path() {
    check(
        r#"
macro_rules! m {
    ($p:path) => { fn foo() { let a = $p; } }
}

m! { foo }

m! { bar::<u8>::baz::<u8> }
"#,
        expect![[r#"
macro_rules! m {
    ($p:path) => { fn foo() { let a = $p; } }
}

fn foo() {
    let a = foo;
}

fn foo() {
    let a = bar::<u8>::baz::<u8> ;
}
"#]],
    );
}

#[test]
fn test_two_paths() {
    check(
        r#"
macro_rules! m {
    ($i:path, $j:path) => { fn foo() { let a = $ i; let b = $j; } }
}
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($i:path, $j:path) => { fn foo() { let a = $ i; let b = $j; } }
}
fn foo() {
    let a = foo;
    let b = bar;
}
"#]],
    );
}

#[test]
fn test_path_with_path() {
    check(
        r#"
macro_rules! m {
    ($p:path) => { fn foo() { let a = $p::bar; } }
}
m! { foo }
"#,
        expect![[r#"
macro_rules! m {
    ($p:path) => { fn foo() { let a = $p::bar; } }
}
fn foo() {
    let a = foo::bar;
}
"#]],
    );
}

#[test]
fn test_expr() {
    check(
        r#"
macro_rules! m {
    ($e:expr) => { fn bar() { $e; } }
}

m! { 2 + 2 * baz(3).quux() }
"#,
        expect![[r#"
macro_rules! m {
    ($e:expr) => { fn bar() { $e; } }
}

fn bar() {
    2+2*baz(3).quux();
}
"#]],
    )
}

#[test]
fn test_last_expr() {
    check(
        r#"
macro_rules! vec {
    ($($item:expr),*) => {{
            let mut v = Vec::new();
            $( v.push($item); )*
            v
    }};
}

fn f() {
    vec![1,2,3];
}
"#,
        expect![[r#"
macro_rules! vec {
    ($($item:expr),*) => {{
            let mut v = Vec::new();
            $( v.push($item); )*
            v
    }};
}

fn f() {
     {
        let mut v = Vec::new();
        v.push(1);
        v.push(2);
        v.push(3);
        v
    };
}
"#]],
    );
}

#[test]
fn test_expr_with_attr() {
    check(
        r#"
macro_rules! m { ($a:expr) => { x!(); } }
m!(#[allow(a)]());
"#,
        expect![[r#"
macro_rules! m { ($a:expr) => { x!(); } }
x!();
"#]],
    )
}

#[test]
fn test_ty() {
    check(
        r#"
macro_rules! foo {
    ($t:ty) => ( fn bar() -> $t {} )
}
foo! { Baz<u8> }
"#,
        expect![[r#"
macro_rules! foo {
    ($t:ty) => ( fn bar() -> $t {} )
}
fn bar() -> Baz<u8> {}
"#]],
    )
}

#[test]
fn test_ty_with_complex_type() {
    check(
        r#"
macro_rules! foo {
    ($t:ty) => ( fn bar() -> $ t {} )
}

foo! { &'a Baz<u8> }

foo! { extern "Rust" fn() -> Ret }
"#,
        expect![[r#"
macro_rules! foo {
    ($t:ty) => ( fn bar() -> $ t {} )
}

fn bar() -> & 'a Baz<u8> {}

fn bar() -> extern"Rust"fn() -> Ret {}
"#]],
    );
}
