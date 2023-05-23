//! Tests for RFC 3086 metavariable expressions.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]
fn test_dollar_dollar() {
    check(
        r#"
macro_rules! register_struct { ($Struct:ident) => {
    macro_rules! register_methods { ($$($method:ident),*) => {
        macro_rules! implement_methods { ($$$$($$val:expr),*) => {
            struct $Struct;
            impl $Struct { $$(fn $method() -> &'static [u32] { &[$$$$($$$$val),*] })*}
        }}
    }}
}}

register_struct!(Foo);
register_methods!(alpha, beta);
implement_methods!(1, 2, 3);
"#,
        expect![[r#"
macro_rules! register_struct { ($Struct:ident) => {
    macro_rules! register_methods { ($$($method:ident),*) => {
        macro_rules! implement_methods { ($$$$($$val:expr),*) => {
            struct $Struct;
            impl $Struct { $$(fn $method() -> &'static [u32] { &[$$$$($$$$val),*] })*}
        }}
    }}
}}

macro_rules !register_methods {
    ($($method: ident), *) = > {
        macro_rules!implement_methods {
            ($$($val: expr), *) = > {
                struct Foo;
                impl Foo {
                    $(fn $method()-> &'static[u32] {
                        &[$$($$val), *]
                    }
                    )*
                }
            }
        }
    }
}
macro_rules !implement_methods {
    ($($val: expr), *) = > {
        struct Foo;
        impl Foo {
            fn alpha()-> &'static[u32] {
                &[$($val), *]
            }
            fn beta()-> &'static[u32] {
                &[$($val), *]
            }
        }
    }
}
struct Foo;
impl Foo {
    fn alpha() -> &'static[u32] {
        &[1, 2, 3]
    }
    fn beta() -> &'static[u32] {
        &[1, 2, 3]
    }
}
"#]],
    )
}

#[test]
fn test_metavar_exprs() {
    check(
        r#"
macro_rules! m {
    ( $( $t:tt )* ) => ( $( ${ignore(t)} -${index()} )-* );
}
const _: i32 = m!(a b c);
    "#,
        expect![[r#"
macro_rules! m {
    ( $( $t:tt )* ) => ( $( ${ignore(t)} -${index()} )-* );
}
const _: i32 = -0--1--2;
    "#]],
    );
}
