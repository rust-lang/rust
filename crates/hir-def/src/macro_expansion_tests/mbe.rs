//! Tests specific to declarative macros, aka macros by example. This covers
//! both stable `macro_rules!` macros as well as unstable `macro` macros.

mod tt_conversion;
mod matching;
mod meta_syntax;
mod metavar_expr;
mod regression;

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn token_mapping_smoke_test() {
    check(
        r#"
// +tokenids
macro_rules! f {
    ( struct $ident:ident ) => {
        struct $ident {
            map: ::std::collections::HashSet<()>,
        }
    };
}

// +tokenids
f!(struct MyTraitMap2);
"#,
        expect![[r##"
// call ids will be shifted by Shift(30)
// +tokenids
macro_rules! f {#0
    (#1 struct#2 $#3ident#4:#5ident#6 )#1 =#7>#8 {#9
        struct#10 $#11ident#12 {#13
            map#14:#15 :#16:#17std#18:#19:#20collections#21:#22:#23HashSet#24<#25(#26)#26>#27,#28
        }#13
    }#9;#29
}#0

// // +tokenids
// f!(struct#1 MyTraitMap2#2);
struct#10 MyTraitMap2#32 {#13
    map#14:#15 ::std#18::collections#21::HashSet#24<#25(#26)#26>#27,#28
}#13
"##]],
    );
}

#[test]
fn token_mapping_floats() {
    // Regression test for https://github.com/rust-lang/rust-analyzer/issues/12216
    // (and related issues)
    check(
        r#"
// +tokenids
macro_rules! f {
    ($($tt:tt)*) => {
        $($tt)*
    };
}

// +tokenids
f! {
    fn main() {
        1;
        1.0;
        let x = 1;
    }
}


"#,
        expect![[r##"
// call ids will be shifted by Shift(18)
// +tokenids
macro_rules! f {#0
    (#1$#2(#3$#4tt#5:#6tt#7)#3*#8)#1 =#9>#10 {#11
        $#12(#13$#14tt#15)#13*#16
    }#11;#17
}#0

// // +tokenids
// f! {
//     fn#1 main#2() {
//         1#5;#6
//         1.0#7;#8
//         let#9 x#10 =#11 1#12;#13
//     }
// }
fn#19 main#20(#21)#21 {#22
    1#23;#24
    1.0#25;#26
    let#27 x#28 =#29 1#30;#31
}#22


"##]],
    );
}

#[test]
fn token_mapping_eager() {
    check(
        r#"
#[rustc_builtin_macro]
#[macro_export]
macro_rules! format_args {}

macro_rules! identity {
    ($expr:expr) => { $expr };
}

fn main(foo: ()) {
    format_args/*+tokenids*/!("{} {} {}", format_args!("{}", 0), foo, identity!(10), "bar")
}

"#,
        expect![[r##"
#[rustc_builtin_macro]
#[macro_export]
macro_rules! format_args {}

macro_rules! identity {
    ($expr:expr) => { $expr };
}

fn main(foo: ()) {
    // format_args/*+tokenids*/!("{} {} {}"#1,#3 format_args!("{}", 0#10),#12 foo#13,#14 identity!(10#18),#21 "bar"#22)
::core#4294967295::fmt#4294967295::Arguments#4294967295::new_v1#4294967295(&#4294967295[#4294967295""#4294967295,#4294967295 " "#4294967295,#4294967295 " "#4294967295,#4294967295 ]#4294967295,#4294967295 &#4294967295[::core#4294967295::fmt#4294967295::ArgumentV1#4294967295::new#4294967295(&#4294967295(::core#4294967295::fmt#4294967295::Arguments#4294967295::new_v1#4294967295(&#4294967295[#4294967295""#4294967295,#4294967295 ]#4294967295,#4294967295 &#4294967295[::core#4294967295::fmt#4294967295::ArgumentV1#4294967295::new#4294967295(&#4294967295(#42949672950#10)#4294967295,#4294967295 ::core#4294967295::fmt#4294967295::Display#4294967295::fmt#4294967295)#4294967295,#4294967295 ]#4294967295)#4294967295)#4294967295,#4294967295 ::core#4294967295::fmt#4294967295::Display#4294967295::fmt#4294967295)#4294967295,#4294967295 ::core#4294967295::fmt#4294967295::ArgumentV1#4294967295::new#4294967295(&#4294967295(#4294967295foo#13)#4294967295,#4294967295 ::core#4294967295::fmt#4294967295::Display#4294967295::fmt#4294967295)#4294967295,#4294967295 ::core#4294967295::fmt#4294967295::ArgumentV1#4294967295::new#4294967295(&#4294967295(#429496729510#18)#4294967295,#4294967295 ::core#4294967295::fmt#4294967295::Display#4294967295::fmt#4294967295)#4294967295,#4294967295 ]#4294967295)#4294967295
}

"##]],
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
        expect![[r##"
macro_rules! m {
    ($m:meta) => ( #[$m] fn bar() {} )
}
#[cfg(target_os = "windows")] fn bar() {}
#[hello::world] fn bar() {}
"##]],
    );
}

#[test]
fn test_meta_doc_comments() {
    cov_mark::check!(test_meta_doc_comments);
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
        expect![[r##"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
#[doc = " Single Line Doc 1"]
#[doc = "\n        MultiLines Doc\n    "] fn bar() {}
"##]],
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
        expect![[r##"
macro_rules! m {
    (#[$m:meta]) => ( #[$m] fn bar() {} )
}
#[doc = concat!("The `", "bla", "` lang item.")] fn bar() {}
"##]],
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
        expect![[r##"
macro_rules! m {
    ($(#[$ m:meta])+) => ( $(#[$m])+ fn bar() {} )
}
#[doc = " 錦瑟無端五十弦，一弦一柱思華年。"]
#[doc = "\n        莊生曉夢迷蝴蝶，望帝春心託杜鵑。\n    "] fn bar() {}
"##]],
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
#[doc = " \\ \" \'"] fn bar() {}
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
        expect![[r##"
macro_rules! m { ($($tt:tt)*) => { abs!(=> $($tt)*); } }
abs!( = > #);
"##]],
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
ok!();
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
/* parse error: expected identifier */
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
//               ERROR@9..10
//                 L_PAREN@9..10 "("
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
