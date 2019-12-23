use super::{infer, type_at, type_at_pos};
use crate::test_db::TestDB;
use insta::assert_snapshot;
use ra_db::fixture::WithFixture;

#[test]
fn cfg_impl_block() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:foo cfg:test
use foo::S as T;
struct S;

#[cfg(test)]
impl S {
    fn foo1(&self) -> i32 { 0 }
}

#[cfg(not(test))]
impl S {
    fn foo2(&self) -> i32 { 0 }
}

fn test() {
    let t = (S.foo1(), S.foo2(), T.foo3(), T.foo4());
    t<|>;
}

//- /foo.rs crate:foo
struct S;

#[cfg(not(test))]
impl S {
    fn foo3(&self) -> i32 { 0 }
}

#[cfg(test)]
impl S {
    fn foo4(&self) -> i32 { 0 }
}
"#,
    );
    assert_eq!("(i32, {unknown}, i32, {unknown})", type_at_pos(&db, pos));
}

#[test]
fn infer_macros_expanded() {
    assert_snapshot!(
        infer(r#"
struct Foo(Vec<i32>);

macro_rules! foo {
    ($($item:expr),*) => {
            {
                Foo(vec![$($item,)*])
            }
    };
}

fn main() {
    let x = foo!(1,2);
}
"#),
        @r###"
    ![0; 17) '{Foo(v...,2,])}': Foo
    ![1; 4) 'Foo': Foo({unknown}) -> Foo
    ![1; 16) 'Foo(vec![1,2,])': Foo
    ![5; 15) 'vec![1,2,]': {unknown}
    [156; 182) '{     ...,2); }': ()
    [166; 167) 'x': Foo
    "###
    );
}

#[test]
fn infer_legacy_textual_scoped_macros_expanded() {
    assert_snapshot!(
        infer(r#"
struct Foo(Vec<i32>);

#[macro_use]
mod m {
    macro_rules! foo {
        ($($item:expr),*) => {
            {
                Foo(vec![$($item,)*])
            }
        };
    }
}

fn main() {
    let x = foo!(1,2);
    let y = crate::foo!(1,2);
}
"#),
        @r###"
    ![0; 17) '{Foo(v...,2,])}': Foo
    ![1; 4) 'Foo': Foo({unknown}) -> Foo
    ![1; 16) 'Foo(vec![1,2,])': Foo
    ![5; 15) 'vec![1,2,]': {unknown}
    [195; 251) '{     ...,2); }': ()
    [205; 206) 'x': Foo
    [228; 229) 'y': {unknown}
    [232; 248) 'crate:...!(1,2)': {unknown}
    "###
    );
}

#[test]
fn infer_path_qualified_macros_expanded() {
    assert_snapshot!(
        infer(r#"
#[macro_export]
macro_rules! foo {
    () => { 42i32 }
}

mod m {
    pub use super::foo as bar;
}

fn main() {
    let x = crate::foo!();
    let y = m::bar!();
}
"#),
        @r###"
    ![0; 5) '42i32': i32
    ![0; 5) '42i32': i32
    [111; 164) '{     ...!(); }': ()
    [121; 122) 'x': i32
    [148; 149) 'y': i32
    "###
    );
}

#[test]
fn infer_type_value_macro_having_same_name() {
    assert_snapshot!(
        infer(r#"
#[macro_export]
macro_rules! foo {
    () => {
        mod foo {
            pub use super::foo;
        }
    };
    ($x:tt) => {
        $x
    };
}

foo!();

fn foo() {
    let foo = foo::foo!(42i32);
}
"#),
        @r###"
    ![0; 5) '42i32': i32
    [171; 206) '{     ...32); }': ()
    [181; 184) 'foo': i32
    "###
    );
}

#[test]
fn processes_impls_generated_by_macros() {
    let t = type_at(
        r#"
//- /main.rs
macro_rules! m {
    ($ident:ident) => (impl Trait for $ident {})
}
trait Trait { fn foo(self) -> u128 {} }
struct S;
m!(S);
fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn infer_impl_items_generated_by_macros() {
    let t = type_at(
        r#"
//- /main.rs
macro_rules! m {
    () => (fn foo(&self) -> u128 {0})
}
struct S;
impl S {
    m!();
}

fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn infer_impl_items_generated_by_macros_chain() {
    let t = type_at(
        r#"
//- /main.rs
macro_rules! m_inner {
    () => {fn foo(&self) -> u128 {0}}
}
macro_rules! m {
    () => {m_inner!();}
}

struct S;
impl S {
    m!();
}

fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn infer_macro_with_dollar_crate_is_correct_in_expr() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:foo
fn test() {
    let x = (foo::foo!(1), foo::foo!(2));
    x<|>;
}

//- /lib.rs crate:foo
#[macro_export]
macro_rules! foo {
    (1) => { $crate::bar!() };
    (2) => { 1 + $crate::baz() };
}

#[macro_export]
macro_rules! bar {
    () => { 42 }
}

pub fn baz() -> usize { 31usize }
"#,
    );
    assert_eq!("(i32, usize)", type_at_pos(&db, pos));
}

#[test]
fn infer_type_value_non_legacy_macro_use_as() {
    assert_snapshot!(
        infer(r#"
mod m {
    macro_rules! _foo {
        ($x:ident) => { type $x = u64; }
    }
    pub(crate) use _foo as foo;
}

m::foo!(foo);
use foo as bar;
fn f() -> bar { 0 }
fn main() {  
    let _a  = f();
}
"#),
        @r###"
        [159; 164) '{ 0 }': u64
        [161; 162) '0': u64
        [175; 199) '{     ...f(); }': ()
        [187; 189) '_a': u64
        [193; 194) 'f': fn f() -> u64
        [193; 196) 'f()': u64        
    "###
    );
}

#[test]
fn infer_builtin_macros_line() {
    assert_snapshot!(
        infer(r#"
#[rustc_builtin_macro]
macro_rules! line {() => {}}

fn main() {
    let x = line!();
}
"#),
        @r###"
    ![0; 1) '6': i32
    [64; 88) '{     ...!(); }': ()
    [74; 75) 'x': i32
    "###
    );
}

#[test]
fn infer_builtin_macros_file() {
    assert_snapshot!(
        infer(r#"
#[rustc_builtin_macro]
macro_rules! file {() => {}}

fn main() {
    let x = file!();
}
"#),
        @r###"
    ![0; 2) '""': &str
    [64; 88) '{     ...!(); }': ()
    [74; 75) 'x': &str
    "###
    );
}

#[test]
fn infer_builtin_macros_column() {
    assert_snapshot!(
        infer(r#"
#[rustc_builtin_macro]
macro_rules! column {() => {}}

fn main() {
    let x = column!();
}
"#),
        @r###"
    ![0; 2) '13': i32
    [66; 92) '{     ...!(); }': ()
    [76; 77) 'x': i32
    "###
    );
}

#[test]
fn infer_derive_clone_simple() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std
#[derive(Clone)]
struct S;
fn test() {
    S.clone()<|>;
}

//- /lib.rs crate:std
#[prelude_import]
use clone::*;
mod clone {
    trait Clone {
        fn clone(&self) -> Self;
    }
}
"#,
    );
    assert_eq!("S", type_at_pos(&db, pos));
}

#[test]
fn infer_derive_clone_with_params() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std
#[derive(Clone)]
struct S;
#[derive(Clone)]
struct Wrapper<T>(T);
struct NonClone;
fn test() {
    (Wrapper(S).clone(), Wrapper(NonClone).clone())<|>;
}

//- /lib.rs crate:std
#[prelude_import]
use clone::*;
mod clone {
    trait Clone {
        fn clone(&self) -> Self;
    }
}
"#,
    );
    assert_eq!("(Wrapper<S>, {unknown})", type_at_pos(&db, pos));
}
