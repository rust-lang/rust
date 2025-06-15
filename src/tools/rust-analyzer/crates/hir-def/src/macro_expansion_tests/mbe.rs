//! Tests specific to declarative macros, aka macros by example. This covers
//! both stable `macro_rules!` macros as well as unstable `macro` macros.

mod matching;
mod meta_syntax;
mod metavar_expr;
mod regression;
mod tt_conversion;

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn token_mapping_smoke_test() {
    check(
        r#"
macro_rules! f {
    ( struct $ident:ident ) => {
        struct $ident {
            map: ::std::collections::HashSet<()>,
        }
    };
}

// +spans+syntaxctxt
f!(struct MyTraitMap2);
"#,
        expect![[r#"
macro_rules! f {
    ( struct $ident:ident ) => {
        struct $ident {
            map: ::std::collections::HashSet<()>,
        }
    };
}

struct#0:MacroRules[8C8E, 0]@58..64#14336# MyTraitMap2#0:MacroCall[D499, 0]@31..42#ROOT2024# {#0:MacroRules[8C8E, 0]@72..73#14336#
    map#0:MacroRules[8C8E, 0]@86..89#14336#:#0:MacroRules[8C8E, 0]@89..90#14336# #0:MacroRules[8C8E, 0]@89..90#14336#::#0:MacroRules[8C8E, 0]@91..93#14336#std#0:MacroRules[8C8E, 0]@93..96#14336#::#0:MacroRules[8C8E, 0]@96..98#14336#collections#0:MacroRules[8C8E, 0]@98..109#14336#::#0:MacroRules[8C8E, 0]@109..111#14336#HashSet#0:MacroRules[8C8E, 0]@111..118#14336#<#0:MacroRules[8C8E, 0]@118..119#14336#(#0:MacroRules[8C8E, 0]@119..120#14336#)#0:MacroRules[8C8E, 0]@120..121#14336#>#0:MacroRules[8C8E, 0]@121..122#14336#,#0:MacroRules[8C8E, 0]@122..123#14336#
}#0:MacroRules[8C8E, 0]@132..133#14336#
"#]],
    );
}

#[test]
fn token_mapping_floats() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/12216
    // (and related issues)
    check(
        r#"
// +spans+syntaxctxt
macro_rules! f {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

// +spans+syntaxctxt
f! {
    fn main() {
        1;
        1.0;
        ((1,),).0.0;
        let x = 1;
    }
}


"#,
        expect![[r#"
// +spans+syntaxctxt
macro_rules! f {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

fn#0:MacroCall[D499, 0]@30..32#ROOT2024# main#0:MacroCall[D499, 0]@33..37#ROOT2024#(#0:MacroCall[D499, 0]@37..38#ROOT2024#)#0:MacroCall[D499, 0]@38..39#ROOT2024# {#0:MacroCall[D499, 0]@40..41#ROOT2024#
    1#0:MacroCall[D499, 0]@50..51#ROOT2024#;#0:MacroCall[D499, 0]@51..52#ROOT2024#
    1.0#0:MacroCall[D499, 0]@61..64#ROOT2024#;#0:MacroCall[D499, 0]@64..65#ROOT2024#
    (#0:MacroCall[D499, 0]@74..75#ROOT2024#(#0:MacroCall[D499, 0]@75..76#ROOT2024#1#0:MacroCall[D499, 0]@76..77#ROOT2024#,#0:MacroCall[D499, 0]@77..78#ROOT2024# )#0:MacroCall[D499, 0]@78..79#ROOT2024#,#0:MacroCall[D499, 0]@79..80#ROOT2024# )#0:MacroCall[D499, 0]@80..81#ROOT2024#.#0:MacroCall[D499, 0]@81..82#ROOT2024#0#0:MacroCall[D499, 0]@82..85#ROOT2024#.#0:MacroCall[D499, 0]@82..85#ROOT2024#0#0:MacroCall[D499, 0]@82..85#ROOT2024#;#0:MacroCall[D499, 0]@85..86#ROOT2024#
    let#0:MacroCall[D499, 0]@95..98#ROOT2024# x#0:MacroCall[D499, 0]@99..100#ROOT2024# =#0:MacroCall[D499, 0]@101..102#ROOT2024# 1#0:MacroCall[D499, 0]@103..104#ROOT2024#;#0:MacroCall[D499, 0]@104..105#ROOT2024#
}#0:MacroCall[D499, 0]@110..111#ROOT2024#


"#]],
    );
}

#[test]
fn eager_expands_with_unresolved_within() {
    check(
        r#"
#[rustc_builtin_macro]
#[macro_export]
macro_rules! concat {}
macro_rules! identity {
    ($tt:tt) => {
        $tt
    }
}

fn main(foo: ()) {
    concat!("hello", identity!("world"), unresolved!(), identity!("!"));
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
#[macro_export]
macro_rules! concat {}
macro_rules! identity {
    ($tt:tt) => {
        $tt
    }
}

fn main(foo: ()) {
    /* error: unresolved macro unresolved */"helloworld!";
}
"##]],
    );
}

#[test]
fn concat_spans() {
    check(
        r#"
#[rustc_builtin_macro]
#[macro_export]
macro_rules! concat {}
macro_rules! identity {
    ($tt:tt) => {
        $tt
    }
}

fn main(foo: ()) {
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat {}
    macro_rules! identity {
        ($tt:tt) => {
            $tt
        }
    }

    fn main(foo: ()) {
        concat/*+spans+syntaxctxt*/!("hello", concat!("w", identity!("o")), identity!("rld"), unresolved!(), identity!("!"));
    }
}

"#,
        expect![[r##"
#[rustc_builtin_macro]
#[macro_export]
macro_rules! concat {}
macro_rules! identity {
    ($tt:tt) => {
        $tt
    }
}

fn main(foo: ()) {
    #[rustc_builtin_macro]
    #[macro_export]
    macro_rules! concat {}
    macro_rules! identity {
        ($tt:tt) => {
            $tt
        }
    }

    fn main(foo: ()) {
        /* error: unresolved macro unresolved */"helloworld!"#0:Fn[B9C7, 0]@236..321#ROOT2024#;
    }
}

"##]],
    );
}

#[test]
fn token_mapping_across_files() {
    check(
        r#"
//- /lib.rs
#[macro_use]
mod foo;

mk_struct/*+spans+syntaxctxt*/!(Foo with u32);
//- /foo.rs
macro_rules! mk_struct {
    ($foo:ident with $ty:ty) => { struct $foo($ty); }
}
"#,
        expect![[r#"
#[macro_use]
mod foo;

struct#1:MacroRules[E572, 0]@59..65#14336# Foo#0:MacroCall[BDD3, 0]@32..35#ROOT2024#(#1:MacroRules[E572, 0]@70..71#14336#u32#0:MacroCall[BDD3, 0]@41..44#ROOT2024#)#1:MacroRules[E572, 0]@74..75#14336#;#1:MacroRules[E572, 0]@75..76#14336#
"#]],
    );
}

#[test]
fn float_field_access_macro_input() {
    check(
        r#"
macro_rules! foo {
    ($expr:expr) => {
        fn foo() {
            $expr;
        }
    };
}
foo!(x .0.1);
foo!(x .2. 3);
foo!(x .4 .5);
"#,
        expect![[r#"
macro_rules! foo {
    ($expr:expr) => {
        fn foo() {
            $expr;
        }
    };
}
fn foo() {
    (x.0.1);
}
fn foo() {
    (x.2.3);
}
fn foo() {
    (x.4.5);
}
"#]],
    );
}

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
    fn from(it: Leaf) -> TokenTree {
        TokenTree::Leaf(it)
    }
}
impl From<Subtree> for TokenTree {
    fn from(it: Subtree) -> TokenTree {
        TokenTree::Subtree(it)
    }
}
"#]],
    );
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
        expect![[r#"
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
"#]],
    );
}

#[test]
fn test_match_group_pattern_with_multiple_defs() {
    // FIXME: The pretty printer breaks by leaving whitespace here, +syntaxctxt is used to avoid that
    check(
        r#"
macro_rules! m {
    ($($i:ident),*) => ( impl Bar { $(fn $i() {})* } );
}
// +syntaxctxt
m! { foo, bar }
"#,
        expect![[r#"
macro_rules! m {
    ($($i:ident),*) => ( impl Bar { $(fn $i() {})* } );
}
impl#\14336# Bar#\14336# {#\14336#
    fn#\14336# foo#\ROOT2024#(#\14336#)#\14336# {#\14336#}#\14336#
    fn#\14336# bar#\ROOT2024#(#\14336#)#\14336# {#\14336#}#\14336#
}#\14336#
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
        expect![[r#"
macro_rules! m {
    ($($i:ident)* #abc) => ( fn baz() { $($i ();)* } );
}
fn baz() {}
"#]],
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
    (1+2)*3;
}
// MACRO_ITEMS@0..17
//   FN@0..17
//     FN_KW@0..2 "fn"
//     NAME@2..5
//       IDENT@2..5 "bar"
//     PARAM_LIST@5..7
//       L_PAREN@5..6 "("
//       R_PAREN@6..7 ")"
//     BLOCK_EXPR@7..17
//       STMT_LIST@7..17
//         L_CURLY@7..8 "{"
//         EXPR_STMT@8..16
//           BIN_EXPR@8..15
//             PAREN_EXPR@8..13
//               L_PAREN@8..9 "("
//               BIN_EXPR@9..12
//                 LITERAL@9..10
//                   INT_NUMBER@9..10 "1"
//                 PLUS@10..11 "+"
//                 LITERAL@11..12
//                   INT_NUMBER@11..12 "2"
//               R_PAREN@12..13 ")"
//             STAR@13..14 "*"
//             LITERAL@14..15
//               INT_NUMBER@14..15 "3"
//           SEMICOLON@15..16 ";"
//         R_CURLY@16..17 "}"

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
    [ $( ( $($i:ident)* ) )* ] => [ ok![$( ( $($i)* ) )*]; ]
}
m! ( (a b) );
"#,
        expect![[r#"
macro_rules! m {
    [ $( ( $($i:ident)* ) )* ] => [ ok![$( ( $($i)* ) )*]; ]
}
ok![(a b)];
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
    field: u32
}
struct Bar {
    field: u32
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
    m!/*+tree*/{}
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
fn test_type_path_is_transcribed_as_expr_path() {
    check(
        r#"
macro_rules! m {
    ($p:path) => { let $p; }
}
fn test() {
    m!(S)
    m!(S<i32>)
    m!(S<S<i32>>)
    m!(S<{ module::CONST < 42 }>)
}
"#,
        expect![[r#"
macro_rules! m {
    ($p:path) => { let $p; }
}
fn test() {
    let S;
    let S:: <i32> ;
    let S:: <S:: <i32>> ;
    let S:: < {
        module::CONST<42
    }
    > ;
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
    (2+2*baz(3).quux());
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
macro_rules! m { ($a:expr) => { ok!(); } }
m!(#[allow(a)]());
"#,
        expect![[r#"
macro_rules! m { ($a:expr) => { ok!(); } }
ok!();
"#]],
    )
}

#[test]
fn test_ty() {
    check(
        r#"
macro_rules! m {
    ($t:ty) => ( fn bar() -> $t {} )
}
m! { Baz<u8> }
"#,
        expect![[r#"
macro_rules! m {
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
macro_rules! m {
    ($t:ty) => ( fn bar() -> $ t {} )
}

m! { &'a Baz<u8> }

m! { extern "Rust" fn() -> Ret }
"#,
        expect![[r#"
macro_rules! m {
    ($t:ty) => ( fn bar() -> $ t {} )
}

fn bar() -> &'a Baz<u8> {}

fn bar() -> extern "Rust" fn() -> Ret {}
"#]],
    );
}

#[test]
fn test_pat_() {
    check(
        r#"
macro_rules! m {
    ($p:pat) => { fn foo() { let $p; } }
}
m! { (a, b) }
"#,
        expect![[r#"
macro_rules! m {
    ($p:pat) => { fn foo() { let $p; } }
}
fn foo() {
    let (a, b);
}
"#]],
    );
}

#[test]
fn test_stmt() {
    check(
        r#"
macro_rules! m {
    ($s:stmt) => ( fn bar() { $s; } )
}
m! { 2 }
m! { let a = 0 }
"#,
        expect![[r#"
macro_rules! m {
    ($s:stmt) => ( fn bar() { $s; } )
}
fn bar() {
    2;
}
fn bar() {
    let a = 0;
}
"#]],
    )
}

#[test]
fn test_single_item() {
    check(
        r#"
macro_rules! m { ($i:item) => ( $i ) }
m! { mod c {} }
"#,
        expect![[r#"
macro_rules! m { ($i:item) => ( $i ) }
mod c {}
"#]],
    )
}

#[test]
fn test_all_items() {
    check(
        r#"
macro_rules! m { ($($i:item)*) => ($($i )*) }
m! {
    extern crate a;
    mod b;
    mod c {}
    use d;
    const E: i32 = 0;
    static F: i32 = 0;
    impl G {}
    struct H;
    enum I { Foo }
    trait J {}
    fn h() {}
    extern {}
    type T = u8;
}
"#,
        expect![[r#"
macro_rules! m { ($($i:item)*) => ($($i )*) }
extern crate a;
mod b;
mod c {}
use d;
const E: i32 = 0;
static F: i32 = 0;
impl G {}
struct H;
enum I {
    Foo
}
trait J {}
fn h() {}
extern {}
type T = u8;
"#]],
    );
}

#[test]
fn test_block() {
    check(
        r#"
macro_rules! m { ($b:block) => { fn foo() $b } }
m! { { 1; } }
"#,
        expect![[r#"
macro_rules! m { ($b:block) => { fn foo() $b } }
fn foo() {
    1;
}
"#]],
    );
}

#[test]
fn test_meta() {
    check(
        r#"
macro_rules! m {
    ($m:meta) => ( #[$m] fn bar() {} )
}
m! { cfg(target_os = "windows") }
m! { hello::world }
"#,
        expect![[r#"
macro_rules! m {
    ($m:meta) => ( #[$m] fn bar() {} )
}
#[cfg(target_os = "windows")] fn bar() {}
#[hello::world] fn bar() {}
"#]],
    );
}

#[test]
fn test_meta_doc_comments() {
    check(
        r#"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
m! {
    /// Single Line Doc 1
    /**
        MultiLines Doc
    */
}
"#,
        expect![[r#"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
#[doc = r" Single Line Doc 1"]
#[doc = r"
        MultiLines Doc
    "] fn bar() {}
"#]],
    );
}

#[test]
fn test_meta_extended_key_value_attributes() {
    check(
        r#"
macro_rules! m {
    (#[$m:meta]) => ( #[$m] fn bar() {} )
}
m! { #[doc = concat!("The `", "bla", "` lang item.")] }
"#,
        expect![[r#"
macro_rules! m {
    (#[$m:meta]) => ( #[$m] fn bar() {} )
}
#[doc = concat!("The `", "bla", "` lang item.")] fn bar() {}
"#]],
    );
}

#[test]
fn test_meta_doc_comments_non_latin() {
    check(
        r#"
macro_rules! m {
    ($(#[$ m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
m! {
    /// 錦瑟無端五十弦，一弦一柱思華年。
    /**
        莊生曉夢迷蝴蝶，望帝春心託杜鵑。
    */
}
"#,
        expect![[r#"
macro_rules! m {
    ($(#[$ m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
#[doc = r" 錦瑟無端五十弦，一弦一柱思華年。"]
#[doc = r"
        莊生曉夢迷蝴蝶，望帝春心託杜鵑。
    "] fn bar() {}
"#]],
    );
}

#[test]
fn test_meta_doc_comments_escaped_characters() {
    check(
        r#"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
m! {
    /// \ " '
}
"#,
        expect![[r##"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
#[doc = r#" \ " '"#] fn bar() {}
"##]],
    );
}

#[test]
fn test_tt_block() {
    check(
        r#"
macro_rules! m { ($tt:tt) => { fn foo() $tt } }
m! { { 1; } }
"#,
        expect![[r#"
macro_rules! m { ($tt:tt) => { fn foo() $tt } }
fn foo() {
    1;
}
"#]],
    );
}

#[test]
fn test_tt_group() {
    check(
        r#"
macro_rules! m { ($($tt:tt)*) => { $($tt)* } }
m! { fn foo() {} }"
"#,
        expect![[r#"
macro_rules! m { ($($tt:tt)*) => { $($tt)* } }
fn foo() {}"
"#]],
    );
}

#[test]
fn test_tt_composite() {
    check(
        r#"
macro_rules! m { ($tt:tt) => { ok!(); } }
m! { => }
m! { = > }
"#,
        expect![[r#"
macro_rules! m { ($tt:tt) => { ok!(); } }
ok!();
/* error: leftover tokens */ok!();
"#]],
    );
}

#[test]
fn test_tt_composite2() {
    check(
        r#"
macro_rules! m { ($($tt:tt)*) => { abs!(=> $($tt)*); } }
m! {#}
"#,
        expect![[r#"
macro_rules! m { ($($tt:tt)*) => { abs!(=> $($tt)*); } }
abs!( = > #);
"#]],
    );
}

#[test]
fn test_tt_with_composite_without_space() {
    // Test macro input without any spaces
    // See https://github.com/rust-lang/rust-analyzer/issues/6692
    check(
        r#"
macro_rules! m { ($ op:tt, $j:path) => ( ok!(); ) }
m!(==,Foo::Bool)
"#,
        expect![[r#"
macro_rules! m { ($ op:tt, $j:path) => ( ok!(); ) }
ok!();
"#]],
    );
}

#[test]
fn test_underscore() {
    check(
        r#"
macro_rules! m { ($_:tt) => { ok!(); } }
m! { => }
"#,
        expect![[r#"
macro_rules! m { ($_:tt) => { ok!(); } }
ok!();
"#]],
    );
}

#[test]
fn test_underscore_not_greedily() {
    check(
        r#"
// `_` overlaps with `$a:ident` but rustc matches it under the `_` token.
macro_rules! m1 {
    ($($a:ident)* _) => { ok!(); }
}
m1![a b c d _];

// `_ => ou` overlaps with `$a:expr => $b:ident` but rustc matches it under `_ => $c:expr`.
macro_rules! m2 {
    ($($a:expr => $b:ident)* _ => $c:expr) => { ok!(); }
}
m2![a => b c => d _ => ou]
"#,
        expect![[r#"
// `_` overlaps with `$a:ident` but rustc matches it under the `_` token.
macro_rules! m1 {
    ($($a:ident)* _) => { ok!(); }
}
ok!();

// `_ => ou` overlaps with `$a:expr => $b:ident` but rustc matches it under `_ => $c:expr`.
macro_rules! m2 {
    ($($a:expr => $b:ident)* _ => $c:expr) => { ok!(); }
}
/* error: unexpected token in input */ok!();
"#]],
    );
}

#[test]
fn test_underscore_flavors() {
    check(
        r#"
macro_rules! m1 { ($a:ty) => { ok!(); } }
m1![_];

macro_rules! m2 { ($a:lifetime) => { ok!(); } }
m2!['_];
"#,
        expect![[r#"
macro_rules! m1 { ($a:ty) => { ok!(); } }
ok!();

macro_rules! m2 { ($a:lifetime) => { ok!(); } }
ok!();
"#]],
    );
}

#[test]
fn test_vertical_bar_with_pat_param() {
    check(
        r#"
macro_rules! m { (|$pat:pat_param| ) => { ok!(); } }
m! { |x| }
 "#,
        expect![[r#"
macro_rules! m { (|$pat:pat_param| ) => { ok!(); } }
ok!();
 "#]],
    );
}

#[test]
fn test_new_std_matches() {
    check(
        //- edition:2021
        r#"
macro_rules! matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {
        match $expression {
            $pattern $(if $guard)? => true,
            _ => false
        }
    };
}
fn main() {
    matches!(0, 0 | 1 if true);
}
 "#,
        expect![[r#"
macro_rules! matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {
        match $expression {
            $pattern $(if $guard)? => true,
            _ => false
        }
    };
}
fn main() {
    match 0 {
        0|1 if true =>true , _=>false
    };
}
 "#]],
    );
}

#[test]
fn test_hygienic_pat() {
    check(
        r#"
//- /new.rs crate:new deps:old edition:2015
old::make!();
fn main() {
    matches!(0, 0 | 1 if true);
}
//- /old.rs crate:old edition:2021
#[macro_export]
macro_rules! make {
    () => {
        macro_rules! matches {
            ($expression:expr, $pattern:pat if $guard:expr ) => {
                match $expression {
                    $pattern if $guard => true,
                    _ => false
                }
            };
        }
    }
}
 "#,
        expect![[r#"
macro_rules !matches {
    ($expression: expr, $pattern: pat if $guard: expr) = > {
        match $expression {
            $pattern if $guard = > true , _ = > false
        }
    }
    ;
}
fn main() {
    match 0 {
        0|1 if true =>true , _=>false
    };
}
"#]],
    );
    check(
        r#"
//- /new.rs crate:new deps:old edition:2021
old::make!();
fn main() {
    matches/*+errors*/!(0, 0 | 1 if true);
}
//- /old.rs crate:old edition:2015
#[macro_export]
macro_rules! make {
    () => {
        macro_rules! matches {
            ($expression:expr, $pattern:pat if $guard:expr ) => {
                match $expression {
                    $pattern if $guard => true,
                    _ => false
                }
            };
        }
    }
}
 "#,
        expect![[r#"
macro_rules !matches {
    ($expression: expr, $pattern: pat if $guard: expr) = > {
        match $expression {
            $pattern if $guard = > true , _ = > false
        }
    }
    ;
}
fn main() {
    /* error: unexpected token in input *//* parse error: expected expression */
/* parse error: expected FAT_ARROW */
/* parse error: expected `,` */
/* parse error: expected pattern */
match 0 {
        0 if $guard=>true , _=>false
    };
}
"#]],
    );
}

#[test]
fn test_dollar_crate_lhs_is_not_meta() {
    check(
        r#"
macro_rules! m {
    ($crate) => { err!(); };
    () => { ok!(); };
}
m!{}
"#,
        expect![[r#"
macro_rules! m {
    ($crate) => { err!(); };
    () => { ok!(); };
}
ok!();
"#]],
    );
}

#[test]
fn test_lifetime() {
    check(
        r#"
macro_rules! m {
    ($lt:lifetime) => { struct Ref<$lt>{ s: &$ lt str } }
}
m! {'a}
"#,
        expect![[r#"
macro_rules! m {
    ($lt:lifetime) => { struct Ref<$lt>{ s: &$ lt str } }
}
struct Ref<'a> {
    s: &'a str
}
"#]],
    );
}

#[test]
fn test_literal() {
    check(
        r#"
macro_rules! m {
    ($type:ty, $lit:literal) => { const VALUE: $type = $ lit; };
}
m!(u8, 0);
"#,
        expect![[r#"
macro_rules! m {
    ($type:ty, $lit:literal) => { const VALUE: $type = $ lit; };
}
const VALUE: u8 = 0;
"#]],
    );

    check(
        r#"
macro_rules! m {
    ($type:ty, $lit:literal) => { const VALUE: $ type = $ lit; };
}
m!(i32, -1);
"#,
        expect![[r#"
macro_rules! m {
    ($type:ty, $lit:literal) => { const VALUE: $ type = $ lit; };
}
const VALUE: i32 = -1;
"#]],
    );
}

#[test]
fn test_boolean_is_ident() {
    check(
        r#"
macro_rules! m {
    ($lit0:literal, $lit1:literal) => { const VALUE: (bool, bool) = ($lit0, $lit1); };
}
m!(true, false);
"#,
        expect![[r#"
macro_rules! m {
    ($lit0:literal, $lit1:literal) => { const VALUE: (bool, bool) = ($lit0, $lit1); };
}
const VALUE: (bool, bool) = (true , false );
"#]],
    );
}

#[test]
fn test_vis() {
    check(
        r#"
macro_rules! m {
    ($vis:vis $name:ident) => { $vis fn $name() {} }
}
m!(pub foo);
m!(foo);
"#,
        expect![[r#"
macro_rules! m {
    ($vis:vis $name:ident) => { $vis fn $name() {} }
}
pub fn foo() {}
fn foo() {}
"#]],
    );
}

#[test]
fn test_inner_macro_rules() {
    check(
        r#"
macro_rules! m {
    ($a:ident, $b:ident, $c:tt) => {
        macro_rules! inner {
            ($bi:ident) => { fn $bi() -> u8 { $c } }
        }

        inner!($a);
        fn $b() -> u8 { $c }
    }
}
m!(x, y, 1);
"#,
        expect![[r#"
macro_rules! m {
    ($a:ident, $b:ident, $c:tt) => {
        macro_rules! inner {
            ($bi:ident) => { fn $bi() -> u8 { $c } }
        }

        inner!($a);
        fn $b() -> u8 { $c }
    }
}
macro_rules !inner {
    ($bi: ident) = > {
        fn $bi()-> u8 {
            1
        }
    }
}
inner!(x);
fn y() -> u8 {
    1
}
"#]],
    );
}

#[test]
fn test_expr_after_path_colons() {
    check(
        r#"
macro_rules! m {
    ($k:expr) => { fn f() { K::$k; } }
}
// +tree +errors
m!(C("0"));
"#,
        expect![[r#"
macro_rules! m {
    ($k:expr) => { fn f() { K::$k; } }
}
/* parse error: expected identifier, `self`, `super`, `crate`, or `Self` */
/* parse error: expected SEMICOLON */
/* parse error: expected SEMICOLON */
/* parse error: expected expression, item or let statement */
fn f() {
    K::(C("0"));
}
// MACRO_ITEMS@0..19
//   FN@0..19
//     FN_KW@0..2 "fn"
//     NAME@2..3
//       IDENT@2..3 "f"
//     PARAM_LIST@3..5
//       L_PAREN@3..4 "("
//       R_PAREN@4..5 ")"
//     BLOCK_EXPR@5..19
//       STMT_LIST@5..19
//         L_CURLY@5..6 "{"
//         EXPR_STMT@6..10
//           PATH_EXPR@6..10
//             PATH@6..10
//               PATH@6..7
//                 PATH_SEGMENT@6..7
//                   NAME_REF@6..7
//                     IDENT@6..7 "K"
//               COLON2@7..9 "::"
//               PATH_SEGMENT@9..10
//                 ERROR@9..10
//                   L_PAREN@9..10 "("
//         EXPR_STMT@10..16
//           CALL_EXPR@10..16
//             PATH_EXPR@10..11
//               PATH@10..11
//                 PATH_SEGMENT@10..11
//                   NAME_REF@10..11
//                     IDENT@10..11 "C"
//             ARG_LIST@11..16
//               L_PAREN@11..12 "("
//               LITERAL@12..15
//                 STRING@12..15 "\"0\""
//               R_PAREN@15..16 ")"
//         ERROR@16..17
//           R_PAREN@16..17 ")"
//         SEMICOLON@17..18 ";"
//         R_CURLY@18..19 "}"

"#]],
    );
}

#[test]
fn test_match_is_not_greedy() {
    check(
        r#"
macro_rules! foo {
    ($($i:ident $(,)*),*) => {};
}
foo!(a,b);
"#,
        expect![[r#"
macro_rules! foo {
    ($($i:ident $(,)*),*) => {};
}

"#]],
    );
}

#[test]
fn expr_interpolation() {
    check(
        r#"
macro_rules! m { ($expr:expr) => { map($expr) } }
fn f() {
    let _ = m!(x + foo);
}
"#,
        expect![[r#"
macro_rules! m { ($expr:expr) => { map($expr) } }
fn f() {
    let _ = map((x+foo));
}
"#]],
    )
}

#[test]
fn mbe_are_not_attributes() {
    check(
        r#"
macro_rules! error {
    () => {struct Bar}
}

#[error]
struct Foo;
"#,
        expect![[r##"
macro_rules! error {
    () => {struct Bar}
}

#[error]
struct Foo;
"##]],
    )
}

#[test]
fn test_punct_without_space() {
    // Puncts are "glued" greedily.
    check(
        r#"
macro_rules! foo {
    (: : :) => { "1 1 1" };
    (: ::) => { "1 2" };
    (:: :) => { "2 1" };

    (: : : :) => { "1 1 1 1" };
    (:: : :) => { "2 1 1" };
    (: :: :) => { "1 2 1" };
    (: : ::) => { "1 1 2" };
    (:: ::) => { "2 2" };
}

fn test() {
    foo!(:::);
    foo!(: :::);
    foo!(::::);
}
"#,
        expect![[r#"
macro_rules! foo {
    (: : :) => { "1 1 1" };
    (: ::) => { "1 2" };
    (:: :) => { "2 1" };

    (: : : :) => { "1 1 1 1" };
    (:: : :) => { "2 1 1" };
    (: :: :) => { "1 2 1" };
    (: : ::) => { "1 1 2" };
    (:: ::) => { "2 2" };
}

fn test() {
    "2 1";
    "1 2 1";
    "2 2";
}
"#]],
    );
}

#[test]
fn test_pat_fragment_eof_17441() {
    check(
        r#"
macro_rules! matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? ) => {
        match $expression {
            $pattern $(if $guard)? => true,
            _ => false
        }
    };
}
fn f() {
    matches!(0, 10..);
    matches!(0, 10.. if true);
}
 "#,
        expect![[r#"
macro_rules! matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? ) => {
        match $expression {
            $pattern $(if $guard)? => true,
            _ => false
        }
    };
}
fn f() {
    match 0 {
        10.. =>true , _=>false
    };
    match 0 {
        10..if true =>true , _=>false
    };
}
 "#]],
    );
}

#[test]
fn test_edition_handling_out() {
    check(
        r#"
//- /main.rs crate:main deps:old edition:2021
macro_rules! r#try {
    ($it:expr) => {
        $it?
    };
}
fn f() {
    old::invoke_bare_try!(0);
}
//- /old.rs crate:old edition:2015
#[macro_export]
macro_rules! invoke_bare_try {
    ($it:expr) => {
        try!($it)
    };
}
 "#,
        expect![[r#"
macro_rules! r#try {
    ($it:expr) => {
        $it?
    };
}
fn f() {
    try!(0);
}
"#]],
    );
}

#[test]
fn test_edition_handling_in() {
    check(
        r#"
//- /main.rs crate:main deps:old edition:2021
fn f() {
    old::parse_try_old!(try!{});
}
//- /old.rs crate:old edition:2015
#[macro_export]
macro_rules! parse_try_old {
    ($it:expr) => {};
}
 "#,
        expect![[r#"
fn f() {
    ;
}
"#]],
    );
}

#[test]
fn semicolon_does_not_glue() {
    check(
        r#"
macro_rules! bug {
    ($id: expr) => {
        true
    };
    ($id: expr; $($attr: ident),*) => {
        true
    };
    ($id: expr; $($attr: ident),*; $norm: expr) => {
        true
    };
    ($id: expr; $($attr: ident),*;; $print: expr) => {
        true
    };
    ($id: expr; $($attr: ident),*; $norm: expr; $print: expr) => {
        true
    };
}
fn f() {
    let _ = bug!(a;;;test);
}
    "#,
        expect![[r#"
macro_rules! bug {
    ($id: expr) => {
        true
    };
    ($id: expr; $($attr: ident),*) => {
        true
    };
    ($id: expr; $($attr: ident),*; $norm: expr) => {
        true
    };
    ($id: expr; $($attr: ident),*;; $print: expr) => {
        true
    };
    ($id: expr; $($attr: ident),*; $norm: expr; $print: expr) => {
        true
    };
}
fn f() {
    let _ = true;
}
    "#]],
    );
}

#[test]
fn lifetime_repeat() {
    check(
        r#"
macro_rules! m {
    ($($x:expr)'a*) => (stringify!($($x)'b*));
}
fn f() {
    let _ = m!(0 'a 1 'a 2);
}
    "#,
        expect![[r#"
macro_rules! m {
    ($($x:expr)'a*) => (stringify!($($x)'b*));
}
fn f() {
    let _ = stringify!(0 'b 1 'b 2);
}
    "#]],
    );
}
