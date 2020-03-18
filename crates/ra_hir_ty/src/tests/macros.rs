use super::{infer, type_at, type_at_pos};
use crate::test_db::TestDB;
use insta::assert_snapshot;
use ra_db::fixture::WithFixture;

#[test]
fn cfg_impl_def() {
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
fn expr_macro_expanded_in_various_places() {
    assert_snapshot!(
        infer(r#"
macro_rules! spam {
    () => (1isize);
}

fn spam() {
    spam!();
    (spam!());
    spam!().spam(spam!());
    for _ in spam!() {}
    || spam!();
    while spam!() {}
    break spam!();
    return spam!();
    match spam!() {
        _ if spam!() => spam!(),
    }
    spam!()(spam!());
    Spam { spam: spam!() };
    spam!()[spam!()];
    await spam!();
    spam!() as usize;
    &spam!();
    -spam!();
    spam!()..spam!();
    spam!() + spam!();
}
"#),
        @r###"
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    ![0; 6) '1isize': isize
    [54; 457) '{     ...!(); }': !
    [88; 109) 'spam!(...am!())': {unknown}
    [115; 134) 'for _ ...!() {}': ()
    [119; 120) '_': {unknown}
    [132; 134) '{}': ()
    [139; 149) '|| spam!()': || -> isize
    [155; 171) 'while ...!() {}': ()
    [169; 171) '{}': ()
    [176; 189) 'break spam!()': !
    [195; 209) 'return spam!()': !
    [215; 269) 'match ...     }': isize
    [239; 240) '_': isize
    [274; 290) 'spam!(...am!())': {unknown}
    [296; 318) 'Spam {...m!() }': {unknown}
    [324; 340) 'spam!(...am!()]': {unknown}
    [365; 381) 'spam!(... usize': usize
    [387; 395) '&spam!()': &isize
    [401; 409) '-spam!()': isize
    [415; 431) 'spam!(...pam!()': {unknown}
    [437; 454) 'spam!(...pam!()': isize
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
        [175; 197) '{     ...f(); }': ()
        [185; 187) '_a': u64
        [191; 192) 'f': fn f() -> u64
        [191; 194) 'f()': u64
    "###
    );
}

#[test]
fn infer_local_macro() {
    assert_snapshot!(
        infer(r#"
fn main() {
    macro_rules! foo {
        () => { 1usize }
    }
    let _a  = foo!();
}
"#),
        @r###"
        ![0; 6) '1usize': usize
        [11; 90) '{     ...!(); }': ()
        [17; 66) 'macro_...     }': {unknown}
        [75; 77) '_a': usize
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
    ![0; 1) '0': i32
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
    ![0; 1) '0': i32
    [66; 92) '{     ...!(); }': ()
    [76; 77) 'x': i32
    "###
    );
}

#[test]
fn infer_builtin_macros_concat() {
    assert_snapshot!(
        infer(r#"
#[rustc_builtin_macro]
macro_rules! concat {() => {}}

fn main() {
    let x = concat!("hello", concat!("world", "!"));
}
"#),
        @r###"
    ![0; 13) '"helloworld!"': &str
    [66; 122) '{     ...")); }': ()
    [76; 77) 'x': &str
    "###
    );
}

#[test]
fn infer_builtin_macros_include() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    bar()<|>;
}

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
    assert_eq!("u32", type_at_pos(&db, pos));
}

#[test]
fn infer_builtin_macros_include_concat() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

include!(concat!("f", "oo.rs"));

fn main() {
    bar()<|>;
}

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
    assert_eq!("u32", type_at_pos(&db, pos));
}

#[test]
fn infer_builtin_macros_include_concat_with_bad_env_should_failed() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

#[rustc_builtin_macro]
macro_rules! env {() => {}}

include!(concat!(env!("OUT_DIR"), "/foo.rs"));

fn main() {
    bar()<|>;
}

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
    assert_eq!("{unknown}", type_at_pos(&db, pos));
}

#[test]
fn infer_builtin_macros_include_itself_should_failed() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("main.rs");

fn main() {
    0<|>
}
"#,
    );
    assert_eq!("i32", type_at_pos(&db, pos));
}

#[test]
fn infer_builtin_macros_concat_with_lazy() {
    assert_snapshot!(
        infer(r#"
macro_rules! hello {() => {"hello"}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

fn main() {
    let x = concat!(hello!(), concat!("world", "!"));
}
"#),
        @r###"
    ![0; 13) '"helloworld!"': &str
    [104; 161) '{     ...")); }': ()
    [114; 115) 'x': &str
    "###
    );
}

#[test]
fn infer_builtin_macros_env() {
    assert_snapshot!(
        infer(r#"
//- /main.rs env:foo=bar
#[rustc_builtin_macro]
macro_rules! env {() => {}}

fn main() {
    let x = env!("foo");
}
"#),
        @r###"
    ![0; 5) '"bar"': &str
    [88; 116) '{     ...o"); }': ()
    [98; 99) 'x': &str
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

#[test]
fn infer_custom_derive_simple() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:foo
use foo::Foo;

#[derive(Foo)]
struct S{}

fn test() {
    S{}<|>;
}

//- /lib.rs crate:foo
#[proc_macro_derive(Foo)]
pub fn derive_foo(_item: TokenStream) -> TokenStream {    
}
"#,
    );
    assert_eq!("S", type_at_pos(&db, pos));
}
