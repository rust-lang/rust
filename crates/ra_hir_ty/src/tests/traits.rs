use insta::assert_snapshot;
use ra_db::fixture::WithFixture;
use test_utils::mark;

use crate::test_db::TestDB;

use super::{infer, infer_with_mismatches, type_at, type_at_pos};

#[test]
fn infer_await() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:core

struct IntFuture;

impl Future for IntFuture {
    type Output = u64;
}

fn test() {
    let r = IntFuture;
    let v = r.await;
    v<|>;
}

//- /core.rs crate:core
#[prelude_import] use future::*;
mod future {
    #[lang = "future_trait"]
    trait Future {
        type Output;
    }
}

"#,
    );
    assert_eq!("u64", type_at_pos(&db, pos));
}

#[test]
fn infer_async() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:core

async fn foo() -> u64 {
    128
}

fn test() {
    let r = foo();
    let v = r.await;
    v<|>;
}

//- /core.rs crate:core
#[prelude_import] use future::*;
mod future {
    #[lang = "future_trait"]
    trait Future {
        type Output;
    }
}

"#,
    );
    assert_eq!("u64", type_at_pos(&db, pos));
}

#[test]
fn infer_desugar_async() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:core

async fn foo() -> u64 {
    128
}

fn test() {
    let r = foo();
    r<|>;
}

//- /core.rs crate:core
#[prelude_import] use future::*;
mod future {
    trait Future {
        type Output;
    }
}

"#,
    );
    assert_eq!("impl Future<Output = u64>", type_at_pos(&db, pos));
}

#[test]
fn infer_try() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:core

fn test() {
    let r: Result<i32, u64> = Result::Ok(1);
    let v = r?;
    v<|>;
}

//- /core.rs crate:core

#[prelude_import] use ops::*;
mod ops {
    trait Try {
        type Ok;
        type Error;
    }
}

#[prelude_import] use result::*;
mod result {
    enum Result<O, E> {
        Ok(O),
        Err(E)
    }

    impl<O, E> crate::ops::Try for Result<O, E> {
        type Ok = O;
        type Error = E;
    }
}

"#,
    );
    assert_eq!("i32", type_at_pos(&db, pos));
}

#[test]
fn infer_for_loop() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:core,alloc

use alloc::collections::Vec;

fn test() {
    let v = Vec::new();
    v.push("foo");
    for x in v {
        x<|>;
    }
}

//- /core.rs crate:core

#[prelude_import] use iter::*;
mod iter {
    trait IntoIterator {
        type Item;
    }
}

//- /alloc.rs crate:alloc deps:core

mod collections {
    struct Vec<T> {}
    impl<T> Vec<T> {
        fn new() -> Self { Vec {} }
        fn push(&mut self, t: T) { }
    }

    impl<T> IntoIterator for Vec<T> {
        type Item=T;
    }
}
"#,
    );
    assert_eq!("&str", type_at_pos(&db, pos));
}

#[test]
fn infer_ops_neg() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std

struct Bar;
struct Foo;

impl std::ops::Neg for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = -a;
    b<|>;
}

//- /std.rs crate:std

#[prelude_import] use ops::*;
mod ops {
    #[lang = "neg"]
    pub trait Neg {
        type Output;
    }
}
"#,
    );
    assert_eq!("Foo", type_at_pos(&db, pos));
}

#[test]
fn infer_ops_not() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std

struct Bar;
struct Foo;

impl std::ops::Not for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = !a;
    b<|>;
}

//- /std.rs crate:std

#[prelude_import] use ops::*;
mod ops {
    #[lang = "not"]
    pub trait Not {
        type Output;
    }
}
"#,
    );
    assert_eq!("Foo", type_at_pos(&db, pos));
}

#[test]
fn infer_from_bound_1() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn foo<T: Trait<u32>>(t: T) {}
fn test() {
    let s = S(unknown);
    foo(s);
}
"#),
        @r###"
    86..87 't': T
    92..94 '{}': ()
    105..144 '{     ...(s); }': ()
    115..116 's': S<u32>
    119..120 'S': S<u32>(u32) -> S<u32>
    119..129 'S(unknown)': S<u32>
    121..128 'unknown': u32
    135..138 'foo': fn foo<S<u32>>(S<u32>)
    135..141 'foo(s)': ()
    139..140 's': S<u32>
    "###
    );
}

#[test]
fn infer_from_bound_2() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn foo<U, T: Trait<U>>(t: T) -> U {}
fn test() {
    let s = S(unknown);
    let x: u32 = foo(s);
}
"#),
        @r###"
    87..88 't': T
    98..100 '{}': ()
    111..163 '{     ...(s); }': ()
    121..122 's': S<u32>
    125..126 'S': S<u32>(u32) -> S<u32>
    125..135 'S(unknown)': S<u32>
    127..134 'unknown': u32
    145..146 'x': u32
    154..157 'foo': fn foo<u32, S<u32>>(S<u32>) -> u32
    154..160 'foo(s)': u32
    158..159 's': S<u32>
    "###
    );
}

#[test]
fn trait_default_method_self_bound_implements_trait() {
    mark::check!(trait_self_implements_self);
    assert_snapshot!(
        infer(r#"
trait Trait {
    fn foo(&self) -> i64;
    fn bar(&self) -> {
        let x = self.foo();
    }
}
"#),
        @r###"
    27..31 'self': &Self
    53..57 'self': &Self
    62..97 '{     ...     }': ()
    76..77 'x': i64
    80..84 'self': &Self
    80..90 'self.foo()': i64
    "###
    );
}

#[test]
fn trait_default_method_self_bound_implements_super_trait() {
    assert_snapshot!(
        infer(r#"
trait SuperTrait {
    fn foo(&self) -> i64;
}
trait Trait: SuperTrait {
    fn bar(&self) -> {
        let x = self.foo();
    }
}
"#),
        @r###"
    32..36 'self': &Self
    86..90 'self': &Self
    95..130 '{     ...     }': ()
    109..110 'x': i64
    113..117 'self': &Self
    113..123 'self.foo()': i64
    "###
    );
}

#[test]
fn infer_project_associated_type() {
    assert_snapshot!(
        infer(r#"
trait Iterable {
   type Item;
}
struct S;
impl Iterable for S { type Item = u32; }
fn test<T: Iterable>() {
    let x: <S as Iterable>::Item = 1;
    let y: <T as Iterable>::Item = no_matter;
    let z: T::Item = no_matter;
    let a: <T>::Item = no_matter;
}
"#),
        @r###"
    108..261 '{     ...ter; }': ()
    118..119 'x': u32
    145..146 '1': u32
    156..157 'y': Iterable::Item<T>
    183..192 'no_matter': Iterable::Item<T>
    202..203 'z': Iterable::Item<T>
    215..224 'no_matter': Iterable::Item<T>
    234..235 'a': Iterable::Item<T>
    249..258 'no_matter': Iterable::Item<T>
    "###
    );
}

#[test]
fn infer_return_associated_type() {
    assert_snapshot!(
        infer(r#"
trait Iterable {
   type Item;
}
struct S;
impl Iterable for S { type Item = u32; }
fn foo1<T: Iterable>(t: T) -> T::Item {}
fn foo2<T: Iterable>(t: T) -> <T as Iterable>::Item {}
fn foo3<T: Iterable>(t: T) -> <T>::Item {}
fn test() {
    let x = foo1(S);
    let y = foo2(S);
    let z = foo3(S);
}
"#),
        @r###"
    106..107 't': T
    123..125 '{}': ()
    147..148 't': T
    178..180 '{}': ()
    202..203 't': T
    221..223 '{}': ()
    234..300 '{     ...(S); }': ()
    244..245 'x': u32
    248..252 'foo1': fn foo1<S>(S) -> <S as Iterable>::Item
    248..255 'foo1(S)': u32
    253..254 'S': S
    265..266 'y': u32
    269..273 'foo2': fn foo2<S>(S) -> <S as Iterable>::Item
    269..276 'foo2(S)': u32
    274..275 'S': S
    286..287 'z': u32
    290..294 'foo3': fn foo3<S>(S) -> <S as Iterable>::Item
    290..297 'foo3(S)': u32
    295..296 'S': S
    "###
    );
}

#[test]
fn infer_associated_type_bound() {
    assert_snapshot!(
        infer(r#"
trait Iterable {
   type Item;
}
fn test<T: Iterable<Item=u32>>() {
    let y: T::Item = unknown;
}
"#),
        @r###"
    67..100 '{     ...own; }': ()
    77..78 'y': u32
    90..97 'unknown': u32
    "###
    );
}

#[test]
fn infer_const_body() {
    assert_snapshot!(
        infer(r#"
const A: u32 = 1 + 1;
static B: u64 = { let x = 1; x };
"#),
        @r###"
    16..17 '1': u32
    16..21 '1 + 1': u32
    20..21 '1': u32
    39..55 '{ let ...1; x }': u64
    45..46 'x': u64
    49..50 '1': u64
    52..53 'x': u64
    "###
    );
}

#[test]
fn tuple_struct_fields() {
    assert_snapshot!(
        infer(r#"
struct S(i32, u64);
fn test() -> u64 {
    let a = S(4, 6);
    let b = a.0;
    a.1
}
"#),
        @r###"
    38..87 '{     ... a.1 }': u64
    48..49 'a': S
    52..53 'S': S(i32, u64) -> S
    52..59 'S(4, 6)': S
    54..55 '4': i32
    57..58 '6': u64
    69..70 'b': i32
    73..74 'a': S
    73..76 'a.0': i32
    82..83 'a': S
    82..85 'a.1': u64
    "###
    );
}

#[test]
fn tuple_struct_with_fn() {
    assert_snapshot!(
        infer(r#"
struct S(fn(u32) -> u64);
fn test() -> u64 {
    let a = S(|i| 2*i);
    let b = a.0(4);
    a.0(2)
}
"#),
        @r###"
    44..102 '{     ...0(2) }': u64
    54..55 'a': S
    58..59 'S': S(fn(u32) -> u64) -> S
    58..68 'S(|i| 2*i)': S
    60..67 '|i| 2*i': |u32| -> u64
    61..62 'i': u32
    64..65 '2': u32
    64..67 '2*i': u32
    66..67 'i': u32
    78..79 'b': u64
    82..83 'a': S
    82..85 'a.0': fn(u32) -> u64
    82..88 'a.0(4)': u64
    86..87 '4': u32
    94..95 'a': S
    94..97 'a.0': fn(u32) -> u64
    94..100 'a.0(2)': u64
    98..99 '2': u32
    "###
    );
}

#[test]
fn indexing_arrays() {
    assert_snapshot!(
        infer("fn main() { &mut [9][2]; }"),
        @r###"
    10..26 '{ &mut...[2]; }': ()
    12..23 '&mut [9][2]': &mut {unknown}
    17..20 '[9]': [i32; _]
    17..23 '[9][2]': {unknown}
    18..19 '9': i32
    21..22 '2': i32
    "###
    )
}

#[test]
fn infer_ops_index() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std

struct Bar;
struct Foo;

impl std::ops::Index<u32> for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = a[1u32];
    b<|>;
}

//- /std.rs crate:std

#[prelude_import] use ops::*;
mod ops {
    #[lang = "index"]
    pub trait Index<Idx> {
        type Output;
    }
}
"#,
    );
    assert_eq!("Foo", type_at_pos(&db, pos));
}

#[test]
fn infer_ops_index_autoderef() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std
fn test() {
    let a = &[1u32, 2, 3];
    let b = a[1u32];
    b<|>;
}

//- /std.rs crate:std
impl<T> ops::Index<u32> for [T] {
    type Output = T;
}

#[prelude_import] use ops::*;
mod ops {
    #[lang = "index"]
    pub trait Index<Idx> {
        type Output;
    }
}
"#,
    );
    assert_eq!("u32", type_at_pos(&db, pos));
}

#[test]
fn deref_trait() {
    let t = type_at(
        r#"
//- /main.rs
#[lang = "deref"]
trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct Arc<T>;
impl<T> Deref for Arc<T> {
    type Target = T;
}

struct S;
impl S {
    fn foo(&self) -> u128 {}
}

fn test(s: Arc<S>) {
    (*s, s.foo())<|>;
}
"#,
    );
    assert_eq!(t, "(S, u128)");
}

#[test]
fn deref_trait_with_inference_var() {
    let t = type_at(
        r#"
//- /main.rs
#[lang = "deref"]
trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct Arc<T>;
fn new_arc<T>() -> Arc<T> {}
impl<T> Deref for Arc<T> {
    type Target = T;
}

struct S;
fn foo(a: Arc<S>) {}

fn test() {
    let a = new_arc();
    let b = (*a)<|>;
    foo(a);
}
"#,
    );
    assert_eq!(t, "S");
}

#[test]
fn deref_trait_infinite_recursion() {
    let t = type_at(
        r#"
//- /main.rs
#[lang = "deref"]
trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct S;

impl Deref for S {
    type Target = S;
}

fn test(s: S) {
    s.foo()<|>;
}
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn deref_trait_with_question_mark_size() {
    let t = type_at(
        r#"
//- /main.rs
#[lang = "deref"]
trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct Arc<T>;
impl<T> Deref for Arc<T> {
    type Target = T;
}

struct S;
impl S {
    fn foo(&self) -> u128 {}
}

fn test(s: Arc<S>) {
    (*s, s.foo())<|>;
}
"#,
    );
    assert_eq!(t, "(S, u128)");
}

#[test]
fn obligation_from_function_clause() {
    let t = type_at(
        r#"
//- /main.rs
struct S;

trait Trait<T> {}
impl Trait<u32> for S {}

fn foo<T: Trait<U>, U>(t: T) -> U {}

fn test(s: S) {
    foo(s)<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn obligation_from_method_clause() {
    let t = type_at(
        r#"
//- /main.rs
struct S;

trait Trait<T> {}
impl Trait<isize> for S {}

struct O;
impl O {
    fn foo<T: Trait<U>, U>(&self, t: T) -> U {}
}

fn test() {
    O.foo(S)<|>;
}
"#,
    );
    assert_eq!(t, "isize");
}

#[test]
fn obligation_from_self_method_clause() {
    let t = type_at(
        r#"
//- /main.rs
struct S;

trait Trait<T> {}
impl Trait<i64> for S {}

impl S {
    fn foo<U>(&self) -> U where Self: Trait<U> {}
}

fn test() {
    S.foo()<|>;
}
"#,
    );
    assert_eq!(t, "i64");
}

#[test]
fn obligation_from_impl_clause() {
    let t = type_at(
        r#"
//- /main.rs
struct S;

trait Trait<T> {}
impl Trait<&str> for S {}

struct O<T>;
impl<U, T: Trait<U>> O<T> {
    fn foo(&self) -> U {}
}

fn test(o: O<S>) {
    o.foo()<|>;
}
"#,
    );
    assert_eq!(t, "&str");
}

#[test]
fn generic_param_env_1() {
    let t = type_at(
        r#"
//- /main.rs
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl Clone for S {}
impl<T> Trait for T where T: Clone {}
fn test<T: Clone>(t: T) { t.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn generic_param_env_1_not_met() {
    let t = type_at(
        r#"
//- /main.rs
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl Clone for S {}
impl<T> Trait for T where T: Clone {}
fn test<T>(t: T) { t.foo()<|>; }
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn generic_param_env_2() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S {}
fn test<T: Trait>(t: T) { t.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn generic_param_env_2_not_met() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S {}
fn test<T>(t: T) { t.foo()<|>; }
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn generic_param_env_deref() {
    let t = type_at(
        r#"
//- /main.rs
#[lang = "deref"]
trait Deref {
    type Target;
}
trait Trait {}
impl<T> Deref for T where T: Trait {
    type Target = i128;
}
fn test<T: Trait>(t: T) { (*t)<|>; }
"#,
    );
    assert_eq!(t, "i128");
}

#[test]
fn associated_type_placeholder() {
    let t = type_at(
        r#"
//- /main.rs
pub trait ApplyL {
    type Out;
}

pub struct RefMutL<T>;

impl<T> ApplyL for RefMutL<T> {
    type Out = <T as ApplyL>::Out;
}

fn test<T: ApplyL>() {
    let y: <RefMutL<T> as ApplyL>::Out = no_matter;
    y<|>;
}
"#,
    );
    // inside the generic function, the associated type gets normalized to a placeholder `ApplL::Out<T>` [https://rust-lang.github.io/rustc-guide/traits/associated-types.html#placeholder-associated-types].
    assert_eq!(t, "ApplyL::Out<T>");
}

#[test]
fn associated_type_placeholder_2() {
    let t = type_at(
        r#"
//- /main.rs
pub trait ApplyL {
    type Out;
}
fn foo<T: ApplyL>(t: T) -> <T as ApplyL>::Out;

fn test<T: ApplyL>(t: T) {
    let y = foo(t);
    y<|>;
}
"#,
    );
    assert_eq!(t, "ApplyL::Out<T>");
}

#[test]
fn argument_impl_trait() {
    assert_snapshot!(
        infer_with_mismatches(r#"
trait Trait<T> {
    fn foo(&self) -> T;
    fn foo2(&self) -> i64;
}
fn bar(x: impl Trait<u16>) {}
struct S<T>(T);
impl<T> Trait<T> for S<T> {}

fn test(x: impl Trait<u64>, y: &impl Trait<u32>) {
    x;
    y;
    let z = S(1);
    bar(z);
    x.foo();
    y.foo();
    z.foo();
    x.foo2();
    y.foo2();
    z.foo2();
}
"#, true),
        @r###"
    30..34 'self': &Self
    55..59 'self': &Self
    78..79 'x': impl Trait<u16>
    98..100 '{}': ()
    155..156 'x': impl Trait<u64>
    175..176 'y': &impl Trait<u32>
    196..324 '{     ...2(); }': ()
    202..203 'x': impl Trait<u64>
    209..210 'y': &impl Trait<u32>
    220..221 'z': S<u16>
    224..225 'S': S<u16>(u16) -> S<u16>
    224..228 'S(1)': S<u16>
    226..227 '1': u16
    234..237 'bar': fn bar(S<u16>)
    234..240 'bar(z)': ()
    238..239 'z': S<u16>
    246..247 'x': impl Trait<u64>
    246..253 'x.foo()': u64
    259..260 'y': &impl Trait<u32>
    259..266 'y.foo()': u32
    272..273 'z': S<u16>
    272..279 'z.foo()': u16
    285..286 'x': impl Trait<u64>
    285..293 'x.foo2()': i64
    299..300 'y': &impl Trait<u32>
    299..307 'y.foo2()': i64
    313..314 'z': S<u16>
    313..321 'z.foo2()': i64
    "###
    );
}

#[test]
fn argument_impl_trait_type_args_1() {
    assert_snapshot!(
        infer_with_mismatches(r#"
trait Trait {}
trait Foo {
    // this function has an implicit Self param, an explicit type param,
    // and an implicit impl Trait param!
    fn bar<T>(x: impl Trait) -> T { loop {} }
}
fn foo<T>(x: impl Trait) -> T { loop {} }
struct S;
impl Trait for S {}
struct F;
impl Foo for F {}

fn test() {
    Foo::bar(S);
    <F as Foo>::bar(S);
    F::bar(S);
    Foo::bar::<u32>(S);
    <F as Foo>::bar::<u32>(S);

    foo(S);
    foo::<u32>(S);
    foo::<u32, i32>(S); // we should ignore the extraneous i32
}
"#, true),
        @r###"
    156..157 'x': impl Trait
    176..187 '{ loop {} }': T
    178..185 'loop {}': !
    183..185 '{}': ()
    200..201 'x': impl Trait
    220..231 '{ loop {} }': T
    222..229 'loop {}': !
    227..229 '{}': ()
    301..510 '{     ... i32 }': ()
    307..315 'Foo::bar': fn bar<{unknown}, {unknown}>(S) -> {unknown}
    307..318 'Foo::bar(S)': {unknown}
    316..317 'S': S
    324..339 '<F as Foo>::bar': fn bar<F, {unknown}>(S) -> {unknown}
    324..342 '<F as ...bar(S)': {unknown}
    340..341 'S': S
    348..354 'F::bar': fn bar<F, {unknown}>(S) -> {unknown}
    348..357 'F::bar(S)': {unknown}
    355..356 'S': S
    363..378 'Foo::bar::<u32>': fn bar<{unknown}, u32>(S) -> u32
    363..381 'Foo::b...32>(S)': u32
    379..380 'S': S
    387..409 '<F as ...:<u32>': fn bar<F, u32>(S) -> u32
    387..412 '<F as ...32>(S)': u32
    410..411 'S': S
    419..422 'foo': fn foo<{unknown}>(S) -> {unknown}
    419..425 'foo(S)': {unknown}
    423..424 'S': S
    431..441 'foo::<u32>': fn foo<u32>(S) -> u32
    431..444 'foo::<u32>(S)': u32
    442..443 'S': S
    450..465 'foo::<u32, i32>': fn foo<u32>(S) -> u32
    450..468 'foo::<...32>(S)': u32
    466..467 'S': S
    "###
    );
}

#[test]
fn argument_impl_trait_type_args_2() {
    assert_snapshot!(
        infer_with_mismatches(r#"
trait Trait {}
struct S;
impl Trait for S {}
struct F<T>;
impl<T> F<T> {
    fn foo<U>(self, x: impl Trait) -> (T, U) { loop {} }
}

fn test() {
    F.foo(S);
    F::<u32>.foo(S);
    F::<u32>.foo::<i32>(S);
    F::<u32>.foo::<i32, u32>(S); // extraneous argument should be ignored
}
"#, true),
        @r###"
    88..92 'self': F<T>
    94..95 'x': impl Trait
    119..130 '{ loop {} }': (T, U)
    121..128 'loop {}': !
    126..128 '{}': ()
    144..284 '{     ...ored }': ()
    150..151 'F': F<{unknown}>
    150..158 'F.foo(S)': ({unknown}, {unknown})
    156..157 'S': S
    164..172 'F::<u32>': F<u32>
    164..179 'F::<u32>.foo(S)': (u32, {unknown})
    177..178 'S': S
    185..193 'F::<u32>': F<u32>
    185..207 'F::<u3...32>(S)': (u32, i32)
    205..206 'S': S
    213..221 'F::<u32>': F<u32>
    213..240 'F::<u3...32>(S)': (u32, i32)
    238..239 'S': S
    "###
    );
}

#[test]
fn argument_impl_trait_to_fn_pointer() {
    assert_snapshot!(
        infer_with_mismatches(r#"
trait Trait {}
fn foo(x: impl Trait) { loop {} }
struct S;
impl Trait for S {}

fn test() {
    let f: fn(S) -> () = foo;
}
"#, true),
        @r###"
    23..24 'x': impl Trait
    38..49 '{ loop {} }': ()
    40..47 'loop {}': !
    45..47 '{}': ()
    91..124 '{     ...foo; }': ()
    101..102 'f': fn(S)
    118..121 'foo': fn foo(S)
    "###
    );
}

#[test]
fn impl_trait() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn foo(&self) -> T;
    fn foo2(&self) -> i64;
}
fn bar() -> impl Trait<u64> {}

fn test(x: impl Trait<u64>, y: &impl Trait<u64>) {
    x;
    y;
    let z = bar();
    x.foo();
    y.foo();
    z.foo();
    x.foo2();
    y.foo2();
    z.foo2();
}
"#),
        @r###"
    30..34 'self': &Self
    55..59 'self': &Self
    99..101 '{}': ()
    111..112 'x': impl Trait<u64>
    131..132 'y': &impl Trait<u64>
    152..269 '{     ...2(); }': ()
    158..159 'x': impl Trait<u64>
    165..166 'y': &impl Trait<u64>
    176..177 'z': impl Trait<u64>
    180..183 'bar': fn bar() -> impl Trait<u64>
    180..185 'bar()': impl Trait<u64>
    191..192 'x': impl Trait<u64>
    191..198 'x.foo()': u64
    204..205 'y': &impl Trait<u64>
    204..211 'y.foo()': u64
    217..218 'z': impl Trait<u64>
    217..224 'z.foo()': u64
    230..231 'x': impl Trait<u64>
    230..238 'x.foo2()': i64
    244..245 'y': &impl Trait<u64>
    244..252 'y.foo2()': i64
    258..259 'z': impl Trait<u64>
    258..266 'z.foo2()': i64
    "###
    );
}

#[test]
fn simple_return_pos_impl_trait() {
    mark::check!(lower_rpit);
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn foo(&self) -> T;
}
fn bar() -> impl Trait<u64> { loop {} }

fn test() {
    let a = bar();
    a.foo();
}
"#),
        @r###"
    30..34 'self': &Self
    72..83 '{ loop {} }': !
    74..81 'loop {}': !
    79..81 '{}': ()
    95..130 '{     ...o(); }': ()
    105..106 'a': impl Trait<u64>
    109..112 'bar': fn bar() -> impl Trait<u64>
    109..114 'bar()': impl Trait<u64>
    120..121 'a': impl Trait<u64>
    120..127 'a.foo()': u64
    "###
    );
}

#[test]
fn more_return_pos_impl_trait() {
    assert_snapshot!(
        infer(r#"
trait Iterator {
    type Item;
    fn next(&mut self) -> Self::Item;
}
trait Trait<T> {
    fn foo(&self) -> T;
}
fn bar() -> (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>) { loop {} }
fn baz<T>(t: T) -> (impl Iterator<Item = impl Trait<T>>, impl Trait<T>) { loop {} }

fn test() {
    let (a, b) = bar();
    a.next().foo();
    b.foo();
    let (c, d) = baz(1u128);
    c.next().foo();
    d.foo();
}
"#),
        @r###"
    50..54 'self': &mut Self
    102..106 'self': &Self
    185..196 '{ loop {} }': ({unknown}, {unknown})
    187..194 'loop {}': !
    192..194 '{}': ()
    207..208 't': T
    269..280 '{ loop {} }': ({unknown}, {unknown})
    271..278 'loop {}': !
    276..278 '{}': ()
    292..414 '{     ...o(); }': ()
    302..308 '(a, b)': (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>)
    303..304 'a': impl Iterator<Item = impl Trait<u32>>
    306..307 'b': impl Trait<u64>
    311..314 'bar': fn bar() -> (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>)
    311..316 'bar()': (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>)
    322..323 'a': impl Iterator<Item = impl Trait<u32>>
    322..330 'a.next()': impl Trait<u32>
    322..336 'a.next().foo()': u32
    342..343 'b': impl Trait<u64>
    342..349 'b.foo()': u64
    359..365 '(c, d)': (impl Iterator<Item = impl Trait<u128>>, impl Trait<u128>)
    360..361 'c': impl Iterator<Item = impl Trait<u128>>
    363..364 'd': impl Trait<u128>
    368..371 'baz': fn baz<u128>(u128) -> (impl Iterator<Item = impl Trait<u128>>, impl Trait<u128>)
    368..378 'baz(1u128)': (impl Iterator<Item = impl Trait<u128>>, impl Trait<u128>)
    372..377 '1u128': u128
    384..385 'c': impl Iterator<Item = impl Trait<u128>>
    384..392 'c.next()': impl Trait<u128>
    384..398 'c.next().foo()': u128
    404..405 'd': impl Trait<u128>
    404..411 'd.foo()': u128
    "###
    );
}

#[test]
fn dyn_trait() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn foo(&self) -> T;
    fn foo2(&self) -> i64;
}
fn bar() -> dyn Trait<u64> {}

fn test(x: dyn Trait<u64>, y: &dyn Trait<u64>) {
    x;
    y;
    let z = bar();
    x.foo();
    y.foo();
    z.foo();
    x.foo2();
    y.foo2();
    z.foo2();
}
"#),
        @r###"
    30..34 'self': &Self
    55..59 'self': &Self
    98..100 '{}': ()
    110..111 'x': dyn Trait<u64>
    129..130 'y': &dyn Trait<u64>
    149..266 '{     ...2(); }': ()
    155..156 'x': dyn Trait<u64>
    162..163 'y': &dyn Trait<u64>
    173..174 'z': dyn Trait<u64>
    177..180 'bar': fn bar() -> dyn Trait<u64>
    177..182 'bar()': dyn Trait<u64>
    188..189 'x': dyn Trait<u64>
    188..195 'x.foo()': u64
    201..202 'y': &dyn Trait<u64>
    201..208 'y.foo()': u64
    214..215 'z': dyn Trait<u64>
    214..221 'z.foo()': u64
    227..228 'x': dyn Trait<u64>
    227..235 'x.foo2()': i64
    241..242 'y': &dyn Trait<u64>
    241..249 'y.foo2()': i64
    255..256 'z': dyn Trait<u64>
    255..263 'z.foo2()': i64
    "###
    );
}

#[test]
fn dyn_trait_in_impl() {
    assert_snapshot!(
        infer(r#"
trait Trait<T, U> {
    fn foo(&self) -> (T, U);
}
struct S<T, U> {}
impl<T, U> S<T, U> {
    fn bar(&self) -> &dyn Trait<T, U> { loop {} }
}
trait Trait2<T, U> {
    fn baz(&self) -> (T, U);
}
impl<T, U> Trait2<T, U> for dyn Trait<T, U> { }

fn test(s: S<u32, i32>) {
    s.bar().baz();
}
"#),
        @r###"
    33..37 'self': &Self
    103..107 'self': &S<T, U>
    129..140 '{ loop {} }': &dyn Trait<T, U>
    131..138 'loop {}': !
    136..138 '{}': ()
    176..180 'self': &Self
    252..253 's': S<u32, i32>
    268..290 '{     ...z(); }': ()
    274..275 's': S<u32, i32>
    274..281 's.bar()': &dyn Trait<u32, i32>
    274..287 's.bar().baz()': (u32, i32)
    "###
    );
}

#[test]
fn dyn_trait_bare() {
    assert_snapshot!(
        infer(r#"
trait Trait {
    fn foo(&self) -> u64;
}
fn bar() -> Trait {}

fn test(x: Trait, y: &Trait) -> u64 {
    x;
    y;
    let z = bar();
    x.foo();
    y.foo();
    z.foo();
}
"#),
        @r###"
    27..31 'self': &Self
    61..63 '{}': ()
    73..74 'x': dyn Trait
    83..84 'y': &dyn Trait
    101..176 '{     ...o(); }': ()
    107..108 'x': dyn Trait
    114..115 'y': &dyn Trait
    125..126 'z': dyn Trait
    129..132 'bar': fn bar() -> dyn Trait
    129..134 'bar()': dyn Trait
    140..141 'x': dyn Trait
    140..147 'x.foo()': u64
    153..154 'y': &dyn Trait
    153..160 'y.foo()': u64
    166..167 'z': dyn Trait
    166..173 'z.foo()': u64
    "###
    );
}

#[test]
fn weird_bounds() {
    assert_snapshot!(
        infer(r#"
trait Trait {}
fn test(a: impl Trait + 'lifetime, b: impl 'lifetime, c: impl (Trait), d: impl ('lifetime), e: impl ?Sized, f: impl Trait + ?Sized) {
}
"#),
        @r###"
    24..25 'a': impl Trait + {error}
    51..52 'b': impl {error}
    70..71 'c': impl Trait
    87..88 'd': impl {error}
    108..109 'e': impl {error}
    124..125 'f': impl Trait + {error}
    148..151 '{ }': ()
    "###
    );
}

#[test]
#[ignore]
fn error_bound_chalk() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait {
    fn foo(&self) -> u32 {}
}

fn test(x: (impl Trait + UnknownTrait)) {
    x.foo()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn assoc_type_bindings() {
    assert_snapshot!(
        infer(r#"
trait Trait {
    type Type;
}

fn get<T: Trait>(t: T) -> <T as Trait>::Type {}
fn get2<U, T: Trait<Type = U>>(t: T) -> U {}
fn set<T: Trait<Type = u64>>(t: T) -> T {t}

struct S<T>;
impl<T> Trait for S<T> { type Type = T; }

fn test<T: Trait<Type = u32>>(x: T, y: impl Trait<Type = i64>) {
    get(x);
    get2(x);
    get(y);
    get2(y);
    get(set(S));
    get2(set(S));
    get2(S::<str>);
}
"#),
        @r###"
    50..51 't': T
    78..80 '{}': ()
    112..113 't': T
    123..125 '{}': ()
    155..156 't': T
    166..169 '{t}': T
    167..168 't': T
    257..258 'x': T
    263..264 'y': impl Trait<Type = i64>
    290..398 '{     ...r>); }': ()
    296..299 'get': fn get<T>(T) -> <T as Trait>::Type
    296..302 'get(x)': u32
    300..301 'x': T
    308..312 'get2': fn get2<u32, T>(T) -> u32
    308..315 'get2(x)': u32
    313..314 'x': T
    321..324 'get': fn get<impl Trait<Type = i64>>(impl Trait<Type = i64>) -> <impl Trait<Type = i64> as Trait>::Type
    321..327 'get(y)': i64
    325..326 'y': impl Trait<Type = i64>
    333..337 'get2': fn get2<i64, impl Trait<Type = i64>>(impl Trait<Type = i64>) -> i64
    333..340 'get2(y)': i64
    338..339 'y': impl Trait<Type = i64>
    346..349 'get': fn get<S<u64>>(S<u64>) -> <S<u64> as Trait>::Type
    346..357 'get(set(S))': u64
    350..353 'set': fn set<S<u64>>(S<u64>) -> S<u64>
    350..356 'set(S)': S<u64>
    354..355 'S': S<u64>
    363..367 'get2': fn get2<u64, S<u64>>(S<u64>) -> u64
    363..375 'get2(set(S))': u64
    368..371 'set': fn set<S<u64>>(S<u64>) -> S<u64>
    368..374 'set(S)': S<u64>
    372..373 'S': S<u64>
    381..385 'get2': fn get2<str, S<str>>(S<str>) -> str
    381..395 'get2(S::<str>)': str
    386..394 'S::<str>': S<str>
    "###
    );
}

#[test]
fn impl_trait_assoc_binding_projection_bug() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std
pub trait Language {
    type Kind;
}
pub enum RustLanguage {}
impl Language for RustLanguage {
    type Kind = SyntaxKind;
}
struct SyntaxNode<L> {}
fn foo() -> impl Iterator<Item = SyntaxNode<RustLanguage>> {}

trait Clone {
    fn clone(&self) -> Self;
}

fn api_walkthrough() {
    for node in foo() {
        node.clone()<|>;
    }
}

//- /std.rs crate:std
#[prelude_import] use iter::*;
mod iter {
    trait IntoIterator {
        type Item;
    }
    trait Iterator {
        type Item;
    }
    impl<T: Iterator> IntoIterator for T {
        type Item = <T as Iterator>::Item;
    }
}
"#,
    );
    assert_eq!("{unknown}", type_at_pos(&db, pos));
}

#[test]
fn projection_eq_within_chalk() {
    assert_snapshot!(
        infer(r#"
trait Trait1 {
    type Type;
}
trait Trait2<T> {
    fn foo(self) -> T;
}
impl<T, U> Trait2<T> for U where U: Trait1<Type = T> {}

fn test<T: Trait1<Type = u32>>(x: T) {
    x.foo();
}
"#),
        @r###"
    62..66 'self': Self
    164..165 'x': T
    170..186 '{     ...o(); }': ()
    176..177 'x': T
    176..183 'x.foo()': u32
    "###
    );
}

#[test]
fn where_clause_trait_in_scope_for_method_resolution() {
    let t = type_at(
        r#"
//- /main.rs
mod foo {
    trait Trait {
        fn foo(&self) -> u32 {}
    }
}

fn test<T: foo::Trait>(x: T) {
    x.foo()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn super_trait_method_resolution() {
    assert_snapshot!(
        infer(r#"
mod foo {
    trait SuperTrait {
        fn foo(&self) -> u32 {}
    }
}
trait Trait1: foo::SuperTrait {}
trait Trait2 where Self: foo::SuperTrait {}

fn test<T: Trait1, U: Trait2>(x: T, y: U) {
    x.foo();
    y.foo();
}
"#),
        @r###"
    50..54 'self': &Self
    63..65 '{}': ()
    182..183 'x': T
    188..189 'y': U
    194..223 '{     ...o(); }': ()
    200..201 'x': T
    200..207 'x.foo()': u32
    213..214 'y': U
    213..220 'y.foo()': u32
    "###
    );
}

#[test]
fn super_trait_impl_trait_method_resolution() {
    assert_snapshot!(
        infer(r#"
mod foo {
    trait SuperTrait {
        fn foo(&self) -> u32 {}
    }
}
trait Trait1: foo::SuperTrait {}

fn test(x: &impl Trait1) {
    x.foo();
}
"#),
        @r###"
    50..54 'self': &Self
    63..65 '{}': ()
    116..117 'x': &impl Trait1
    133..149 '{     ...o(); }': ()
    139..140 'x': &impl Trait1
    139..146 'x.foo()': u32
    "###
    );
}

#[test]
fn super_trait_cycle() {
    // This just needs to not crash
    assert_snapshot!(
        infer(r#"
trait A: B {}
trait B: A {}

fn test<T: A>(x: T) {
    x.foo();
}
"#),
        @r###"
    44..45 'x': T
    50..66 '{     ...o(); }': ()
    56..57 'x': T
    56..63 'x.foo()': {unknown}
    "###
    );
}

#[test]
fn super_trait_assoc_type_bounds() {
    assert_snapshot!(
        infer(r#"
trait SuperTrait { type Type; }
trait Trait where Self: SuperTrait {}

fn get2<U, T: Trait<Type = U>>(t: T) -> U {}
fn set<T: Trait<Type = u64>>(t: T) -> T {t}

struct S<T>;
impl<T> SuperTrait for S<T> { type Type = T; }
impl<T> Trait for S<T> {}

fn test() {
    get2(set(S));
}
"#),
        @r###"
    103..104 't': T
    114..116 '{}': ()
    146..147 't': T
    157..160 '{t}': T
    158..159 't': T
    259..280 '{     ...S)); }': ()
    265..269 'get2': fn get2<u64, S<u64>>(S<u64>) -> u64
    265..277 'get2(set(S))': u64
    270..273 'set': fn set<S<u64>>(S<u64>) -> S<u64>
    270..276 'set(S)': S<u64>
    274..275 'S': S<u64>
    "###
    );
}

#[test]
fn fn_trait() {
    assert_snapshot!(
        infer(r#"
trait FnOnce<Args> {
    type Output;

    fn call_once(self, args: Args) -> <Self as FnOnce<Args>>::Output;
}

fn test<F: FnOnce(u32, u64) -> u128>(f: F) {
    f.call_once((1, 2));
}
"#),
        @r###"
    57..61 'self': Self
    63..67 'args': Args
    150..151 'f': F
    156..184 '{     ...2)); }': ()
    162..163 'f': F
    162..181 'f.call...1, 2))': u128
    174..180 '(1, 2)': (u32, u64)
    175..176 '1': u32
    178..179 '2': u64
    "###
    );
}

#[test]
fn fn_ptr_and_item() {
    assert_snapshot!(
        infer(r#"
#[lang="fn_once"]
trait FnOnce<Args> {
    type Output;

    fn call_once(self, args: Args) -> Self::Output;
}

trait Foo<T> {
    fn foo(&self) -> T;
}

struct Bar<T>(T);

impl<A1, R, F: FnOnce(A1) -> R> Foo<(A1, R)> for Bar<F> {
    fn foo(&self) -> (A1, R) {}
}

enum Opt<T> { None, Some(T) }
impl<T> Opt<T> {
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Opt<U> {}
}

fn test() {
    let bar: Bar<fn(u8) -> u32>;
    bar.foo();

    let opt: Opt<u8>;
    let f: fn(u8) -> u32;
    opt.map(f);
}
"#),
        @r###"
75..79 'self': Self
81..85 'args': Args
140..144 'self': &Self
244..248 'self': &Bar<F>
261..263 '{}': ()
347..351 'self': Opt<T>
353..354 'f': F
369..371 '{}': ()
385..501 '{     ...(f); }': ()
395..398 'bar': Bar<fn(u8) -> u32>
424..427 'bar': Bar<fn(u8) -> u32>
424..433 'bar.foo()': {unknown}
444..447 'opt': Opt<u8>
466..467 'f': fn(u8) -> u32
488..491 'opt': Opt<u8>
488..498 'opt.map(f)': Opt<FnOnce::Output<fn(u8) -> u32, (u8,)>>
496..497 'f': fn(u8) -> u32
"###
    );
}

#[test]
fn fn_trait_deref_with_ty_default() {
    assert_snapshot!(
        infer(r#"
#[lang = "deref"]
trait Deref {
    type Target;

    fn deref(&self) -> &Self::Target;
}

#[lang="fn_once"]
trait FnOnce<Args> {
    type Output;

    fn call_once(self, args: Args) -> Self::Output;
}

struct Foo;

impl Foo {
    fn foo(&self) -> usize {}
}

struct Lazy<T, F = fn() -> T>(F);

impl<T, F> Lazy<T, F> {
    pub fn new(f: F) -> Lazy<T, F> {}
}

impl<T, F: FnOnce() -> T> Deref for Lazy<T, F> {
    type Target = T;
}

fn test() {
    let lazy1: Lazy<Foo, _> = Lazy::new(|| Foo);
    let r1 = lazy1.foo();

    fn make_foo_fn() -> Foo {}
    let make_foo_fn_ptr: fn() -> Foo = make_foo_fn;
    let lazy2: Lazy<Foo, _> = Lazy::new(make_foo_fn_ptr);
    let r2 = lazy2.foo();
}
"#),
        @r###"
    65..69 'self': &Self
    166..170 'self': Self
    172..176 'args': Args
    240..244 'self': &Foo
    255..257 '{}': ()
    335..336 'f': F
    355..357 '{}': ()
    444..690 '{     ...o(); }': ()
    454..459 'lazy1': Lazy<Foo, || -> Foo>
    476..485 'Lazy::new': fn new<Foo, || -> Foo>(|| -> Foo) -> Lazy<Foo, || -> Foo>
    476..493 'Lazy::...| Foo)': Lazy<Foo, || -> Foo>
    486..492 '|| Foo': || -> Foo
    489..492 'Foo': Foo
    503..505 'r1': usize
    508..513 'lazy1': Lazy<Foo, || -> Foo>
    508..519 'lazy1.foo()': usize
    561..576 'make_foo_fn_ptr': fn() -> Foo
    592..603 'make_foo_fn': fn make_foo_fn() -> Foo
    613..618 'lazy2': Lazy<Foo, fn() -> Foo>
    635..644 'Lazy::new': fn new<Foo, fn() -> Foo>(fn() -> Foo) -> Lazy<Foo, fn() -> Foo>
    635..661 'Lazy::...n_ptr)': Lazy<Foo, fn() -> Foo>
    645..660 'make_foo_fn_ptr': fn() -> Foo
    671..673 'r2': {unknown}
    676..681 'lazy2': Lazy<Foo, fn() -> Foo>
    676..687 'lazy2.foo()': {unknown}
    550..552 '{}': ()
    "###
    );
}

#[test]
fn closure_1() {
    assert_snapshot!(
        infer(r#"
#[lang = "fn_once"]
trait FnOnce<Args> {
    type Output;
}

enum Option<T> { Some(T), None }
impl<T> Option<T> {
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U> {}
}

fn test() {
    let x = Option::Some(1u32);
    x.map(|v| v + 1);
    x.map(|_v| 1u64);
    let y: Option<i64> = x.map(|_v| 1);
}
"#),
        @r###"
    148..152 'self': Option<T>
    154..155 'f': F
    173..175 '{}': ()
    189..308 '{     ... 1); }': ()
    199..200 'x': Option<u32>
    203..215 'Option::Some': Some<u32>(u32) -> Option<u32>
    203..221 'Option...(1u32)': Option<u32>
    216..220 '1u32': u32
    227..228 'x': Option<u32>
    227..243 'x.map(...v + 1)': Option<u32>
    233..242 '|v| v + 1': |u32| -> u32
    234..235 'v': u32
    237..238 'v': u32
    237..242 'v + 1': u32
    241..242 '1': u32
    249..250 'x': Option<u32>
    249..265 'x.map(... 1u64)': Option<u64>
    255..264 '|_v| 1u64': |u32| -> u64
    256..258 '_v': u32
    260..264 '1u64': u64
    275..276 'y': Option<i64>
    292..293 'x': Option<u32>
    292..305 'x.map(|_v| 1)': Option<i64>
    298..304 '|_v| 1': |u32| -> i64
    299..301 '_v': u32
    303..304 '1': i64
    "###
    );
}

#[test]
fn closure_2() {
    assert_snapshot!(
        infer(r#"
trait FnOnce<Args> {
    type Output;
}

fn test<F: FnOnce(u32) -> u64>(f: F) {
    f(1);
    let g = |v| v + 1;
    g(1u64);
    let h = |v| 1u128 + v;
}
"#),
        @r###"
    73..74 'f': F
    79..155 '{     ...+ v; }': ()
    85..86 'f': F
    85..89 'f(1)': {unknown}
    87..88 '1': i32
    99..100 'g': |u64| -> i32
    103..112 '|v| v + 1': |u64| -> i32
    104..105 'v': u64
    107..108 'v': u64
    107..112 'v + 1': i32
    111..112 '1': i32
    118..119 'g': |u64| -> i32
    118..125 'g(1u64)': i32
    120..124 '1u64': u64
    135..136 'h': |u128| -> u128
    139..152 '|v| 1u128 + v': |u128| -> u128
    140..141 'v': u128
    143..148 '1u128': u128
    143..152 '1u128 + v': u128
    151..152 'v': u128
    "###
    );
}

#[test]
fn closure_as_argument_inference_order() {
    assert_snapshot!(
        infer(r#"
#[lang = "fn_once"]
trait FnOnce<Args> {
    type Output;
}

fn foo1<T, U, F: FnOnce(T) -> U>(x: T, f: F) -> U {}
fn foo2<T, U, F: FnOnce(T) -> U>(f: F, x: T) -> U {}

struct S;
impl S {
    fn method(self) -> u64;

    fn foo1<T, U, F: FnOnce(T) -> U>(self, x: T, f: F) -> U {}
    fn foo2<T, U, F: FnOnce(T) -> U>(self, f: F, x: T) -> U {}
}

fn test() {
    let x1 = foo1(S, |s| s.method());
    let x2 = foo2(|s| s.method(), S);
    let x3 = S.foo1(S, |s| s.method());
    let x4 = S.foo2(|s| s.method(), S);
}
"#),
        @r###"
    95..96 'x': T
    101..102 'f': F
    112..114 '{}': ()
    148..149 'f': F
    154..155 'x': T
    165..167 '{}': ()
    202..206 'self': S
    254..258 'self': S
    260..261 'x': T
    266..267 'f': F
    277..279 '{}': ()
    317..321 'self': S
    323..324 'f': F
    329..330 'x': T
    340..342 '{}': ()
    356..515 '{     ... S); }': ()
    366..368 'x1': u64
    371..375 'foo1': fn foo1<S, u64, |S| -> u64>(S, |S| -> u64) -> u64
    371..394 'foo1(S...hod())': u64
    376..377 'S': S
    379..393 '|s| s.method()': |S| -> u64
    380..381 's': S
    383..384 's': S
    383..393 's.method()': u64
    404..406 'x2': u64
    409..413 'foo2': fn foo2<S, u64, |S| -> u64>(|S| -> u64, S) -> u64
    409..432 'foo2(|...(), S)': u64
    414..428 '|s| s.method()': |S| -> u64
    415..416 's': S
    418..419 's': S
    418..428 's.method()': u64
    430..431 'S': S
    442..444 'x3': u64
    447..448 'S': S
    447..472 'S.foo1...hod())': u64
    454..455 'S': S
    457..471 '|s| s.method()': |S| -> u64
    458..459 's': S
    461..462 's': S
    461..471 's.method()': u64
    482..484 'x4': u64
    487..488 'S': S
    487..512 'S.foo2...(), S)': u64
    494..508 '|s| s.method()': |S| -> u64
    495..496 's': S
    498..499 's': S
    498..508 's.method()': u64
    510..511 'S': S
    "###
    );
}

#[test]
fn unselected_projection_in_trait_env_1() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait {
    type Item;
}

trait Trait2 {
    fn foo(&self) -> u32;
}

fn test<T: Trait>() where T::Item: Trait2 {
    let x: T::Item = no_matter;
    x.foo()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn unselected_projection_in_trait_env_2() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait<T> {
    type Item;
}

trait Trait2 {
    fn foo(&self) -> u32;
}

fn test<T, U>() where T::Item: Trait2, T: Trait<U::Item>, U: Trait<()> {
    let x: T::Item = no_matter;
    x.foo()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn unselected_projection_on_impl_self() {
    assert_snapshot!(infer(
        r#"
//- /main.rs
trait Trait {
    type Item;

    fn f(&self, x: Self::Item);
}

struct S;

impl Trait for S {
    type Item = u32;
    fn f(&self, x: Self::Item) { let y = x; }
}

struct S2;

impl Trait for S2 {
    type Item = i32;
    fn f(&self, x: <Self>::Item) { let y = x; }
}
"#,
    ), @r###"
    54..58 'self': &Self
    60..61 'x': Trait::Item<Self>
    140..144 'self': &S
    146..147 'x': u32
    161..175 '{ let y = x; }': ()
    167..168 'y': u32
    171..172 'x': u32
    242..246 'self': &S2
    248..249 'x': i32
    265..279 '{ let y = x; }': ()
    271..272 'y': i32
    275..276 'x': i32
    "###);
}

#[test]
fn unselected_projection_on_trait_self() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait {
    type Item;

    fn f(&self) -> Self::Item { loop {} }
}

struct S;
impl Trait for S {
    type Item = u32;
}

fn test() {
    S.f()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn unselected_projection_chalk_fold() {
    let t = type_at(
        r#"
//- /main.rs
trait Interner {}
trait Fold<I: Interner, TI = I> {
    type Result;
}

struct Ty<I: Interner> {}
impl<I: Interner, TI: Interner> Fold<I, TI> for Ty<I> {
    type Result = Ty<TI>;
}

fn fold<I: Interner, T>(interner: &I, t: T) -> T::Result
where
    T: Fold<I, I>,
{
    loop {}
}

fn foo<I: Interner>(interner: &I, t: Ty<I>) {
    fold(interner, t)<|>;
}
"#,
    );
    assert_eq!(t, "Ty<I>");
}

#[test]
fn trait_impl_self_ty() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait<T> {
   fn foo(&self);
}

struct S;

impl Trait<Self> for S {}

fn test() {
    S.foo()<|>;
}
"#,
    );
    assert_eq!(t, "()");
}

#[test]
fn trait_impl_self_ty_cycle() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait {
   fn foo(&self);
}

struct S<T>;

impl Trait for S<Self> {}

fn test() {
    S.foo()<|>;
}
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn unselected_projection_in_trait_env_cycle_1() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait {
    type Item;
}

trait Trait2<T> {}

fn test<T: Trait>() where T: Trait2<T::Item> {
    let x: T::Item = no_matter<|>;
}
"#,
    );
    // this is a legitimate cycle
    assert_eq!(t, "{unknown}");
}

#[test]
fn unselected_projection_in_trait_env_cycle_2() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait<T> {
    type Item;
}

fn test<T, U>() where T: Trait<U::Item>, U: Trait<T::Item> {
    let x: T::Item = no_matter<|>;
}
"#,
    );
    // this is a legitimate cycle
    assert_eq!(t, "{unknown}");
}

#[test]
fn inline_assoc_type_bounds_1() {
    let t = type_at(
        r#"
//- /main.rs
trait Iterator {
    type Item;
}
trait OtherTrait<T> {
    fn foo(&self) -> T;
}

// workaround for Chalk assoc type normalization problems
pub struct S<T>;
impl<T: Iterator> Iterator for S<T> {
    type Item = <T as Iterator>::Item;
}

fn test<I: Iterator<Item: OtherTrait<u32>>>() {
    let x: <S<I> as Iterator>::Item;
    x.foo()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn inline_assoc_type_bounds_2() {
    let t = type_at(
        r#"
//- /main.rs
trait Iterator {
    type Item;
}

fn test<I: Iterator<Item: Iterator<Item = u32>>>() {
    let x: <<I as Iterator>::Item as Iterator>::Item;
    x<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn proc_macro_server_types() {
    assert_snapshot!(
        infer(r#"
macro_rules! with_api {
    ($S:ident, $self:ident, $m:ident) => {
        $m! {
            TokenStream {
                fn new() -> $S::TokenStream;
            },
            Group {
            },
        }
    };
}
macro_rules! associated_item {
    (type TokenStream) =>
        (type TokenStream: 'static;);
    (type Group) =>
        (type Group: 'static;);
    ($($item:tt)*) => ($($item)*;)
}
macro_rules! declare_server_traits {
    ($($name:ident {
        $(fn $method:ident($($arg:ident: $arg_ty:ty),* $(,)?) $(-> $ret_ty:ty)?;)*
    }),* $(,)?) => {
        pub trait Types {
            $(associated_item!(type $name);)*
        }

        $(pub trait $name: Types {
            $(associated_item!(fn $method($($arg: $arg_ty),*) $(-> $ret_ty)?);)*
        })*

        pub trait Server: Types $(+ $name)* {}
        impl<S: Types $(+ $name)*> Server for S {}
    }
}

with_api!(Self, self_, declare_server_traits);
struct G {}
struct T {}
struct Rustc;
impl Types for Rustc {
    type TokenStream = T;
    type Group = G;
}

fn make<T>() -> T { loop {} }
impl TokenStream for Rustc {
    fn new() -> Self::TokenStream {
        let group: Self::Group = make();
        make()
    }
}
"#),
        @r###"
    1062..1073 '{ loop {} }': T
    1064..1071 'loop {}': !
    1069..1071 '{}': ()
    1137..1200 '{     ...     }': T
    1151..1156 'group': G
    1172..1176 'make': fn make<G>() -> G
    1172..1178 'make()': G
    1188..1192 'make': fn make<T>() -> T
    1188..1194 'make()': T
    "###
    );
}

#[test]
fn unify_impl_trait() {
    assert_snapshot!(
        infer_with_mismatches(r#"
trait Trait<T> {}

fn foo(x: impl Trait<u32>) { loop {} }
fn bar<T>(x: impl Trait<T>) -> T { loop {} }

struct S<T>(T);
impl<T> Trait<T> for S<T> {}

fn default<T>() -> T { loop {} }

fn test() -> impl Trait<i32> {
    let s1 = S(default());
    foo(s1);
    let x: i32 = bar(S(default()));
    S(default())
}
"#, true),
        @r###"
    27..28 'x': impl Trait<u32>
    47..58 '{ loop {} }': ()
    49..56 'loop {}': !
    54..56 '{}': ()
    69..70 'x': impl Trait<T>
    92..103 '{ loop {} }': T
    94..101 'loop {}': !
    99..101 '{}': ()
    172..183 '{ loop {} }': T
    174..181 'loop {}': !
    179..181 '{}': ()
    214..310 '{     ...t()) }': S<{unknown}>
    224..226 's1': S<u32>
    229..230 'S': S<u32>(u32) -> S<u32>
    229..241 'S(default())': S<u32>
    231..238 'default': fn default<u32>() -> u32
    231..240 'default()': u32
    247..250 'foo': fn foo(S<u32>)
    247..254 'foo(s1)': ()
    251..253 's1': S<u32>
    264..265 'x': i32
    273..276 'bar': fn bar<i32>(S<i32>) -> i32
    273..290 'bar(S(...lt()))': i32
    277..278 'S': S<i32>(i32) -> S<i32>
    277..289 'S(default())': S<i32>
    279..286 'default': fn default<i32>() -> i32
    279..288 'default()': i32
    296..297 'S': S<{unknown}>({unknown}) -> S<{unknown}>
    296..308 'S(default())': S<{unknown}>
    298..305 'default': fn default<{unknown}>() -> {unknown}
    298..307 'default()': {unknown}
    "###
    );
}

#[test]
fn assoc_types_from_bounds() {
    assert_snapshot!(
        infer(r#"
//- /main.rs
#[lang = "fn_once"]
trait FnOnce<Args> {
    type Output;
}

trait T {
    type O;
}

impl T for () {
    type O = ();
}

fn f<X, F>(_v: F)
where
    X: T,
    F: FnOnce(&X::O),
{ }

fn main() {
    f::<(), _>(|z| { z; });
}
"#),
        @r###"
    147..149 '_v': F
    192..195 '{ }': ()
    207..238 '{     ... }); }': ()
    213..223 'f::<(), _>': fn f<(), |&()| -> ()>(|&()| -> ())
    213..235 'f::<()... z; })': ()
    224..234 '|z| { z; }': |&()| -> ()
    225..226 'z': &()
    228..234 '{ z; }': ()
    230..231 'z': &()
    "###
    );
}

#[test]
fn associated_type_bound() {
    let t = type_at(
        r#"
//- /main.rs
pub trait Trait {
    type Item: OtherTrait<u32>;
}
pub trait OtherTrait<T> {
    fn foo(&self) -> T;
}

// this is just a workaround for chalk#234
pub struct S<T>;
impl<T: Trait> Trait for S<T> {
    type Item = <T as Trait>::Item;
}

fn test<T: Trait>() {
    let y: <S<T> as Trait>::Item = no_matter;
    y.foo()<|>;
}
"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn dyn_trait_through_chalk() {
    let t = type_at(
        r#"
//- /main.rs
struct Box<T> {}
#[lang = "deref"]
trait Deref {
    type Target;
}
impl<T> Deref for Box<T> {
    type Target = T;
}
trait Trait {
    fn foo(&self);
}

fn test(x: Box<dyn Trait>) {
    x.foo()<|>;
}
"#,
    );
    assert_eq!(t, "()");
}

#[test]
fn string_to_owned() {
    let t = type_at(
        r#"
//- /main.rs
struct String {}
pub trait ToOwned {
    type Owned;
    fn to_owned(&self) -> Self::Owned;
}
impl ToOwned for str {
    type Owned = String;
}
fn test() {
    "foo".to_owned()<|>;
}
"#,
    );
    assert_eq!(t, "String");
}

#[test]
fn iterator_chain() {
    assert_snapshot!(
        infer(r#"
//- /main.rs
#[lang = "fn_once"]
trait FnOnce<Args> {
    type Output;
}
#[lang = "fn_mut"]
trait FnMut<Args>: FnOnce<Args> { }

enum Option<T> { Some(T), None }
use Option::*;

pub trait Iterator {
    type Item;

    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F>
    where
        F: FnMut(Self::Item) -> Option<B>,
    { loop {} }

    fn for_each<F>(self, f: F)
    where
        F: FnMut(Self::Item),
    { loop {} }
}

pub trait IntoIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
    fn into_iter(self) -> Self::IntoIter;
}

pub struct FilterMap<I, F> { }
impl<B, I: Iterator, F> Iterator for FilterMap<I, F>
where
    F: FnMut(I::Item) -> Option<B>,
{
    type Item = B;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<I: Iterator> IntoIterator for I {
    type Item = I::Item;
    type IntoIter = I;

    fn into_iter(self) -> I {
        self
    }
}

struct Vec<T> {}
impl<T> Vec<T> {
    fn new() -> Self { loop {} }
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;
}

pub struct IntoIter<T> { }
impl<T> Iterator for IntoIter<T> {
    type Item = T;
}

fn main() {
    Vec::<i32>::new().into_iter()
      .filter_map(|x| if x > 0 { Some(x as u32) } else { None })
      .for_each(|y| { y; });
}
"#),
        @r###"
    240..244 'self': Self
    246..247 'f': F
    331..342 '{ loop {} }': FilterMap<Self, F>
    333..340 'loop {}': !
    338..340 '{}': ()
    363..367 'self': Self
    369..370 'f': F
    419..430 '{ loop {} }': ()
    421..428 'loop {}': !
    426..428 '{}': ()
    539..543 'self': Self
    868..872 'self': I
    879..899 '{     ...     }': I
    889..893 'self': I
    958..969 '{ loop {} }': Vec<T>
    960..967 'loop {}': !
    965..967 '{}': ()
    1156..1287 '{     ... }); }': ()
    1162..1177 'Vec::<i32>::new': fn new<i32>() -> Vec<i32>
    1162..1179 'Vec::<...:new()': Vec<i32>
    1162..1191 'Vec::<...iter()': IntoIter<i32>
    1162..1256 'Vec::<...one })': FilterMap<IntoIter<i32>, |i32| -> Option<u32>>
    1162..1284 'Vec::<... y; })': ()
    1210..1255 '|x| if...None }': |i32| -> Option<u32>
    1211..1212 'x': i32
    1214..1255 'if x >...None }': Option<u32>
    1217..1218 'x': i32
    1217..1222 'x > 0': bool
    1221..1222 '0': i32
    1223..1241 '{ Some...u32) }': Option<u32>
    1225..1229 'Some': Some<u32>(u32) -> Option<u32>
    1225..1239 'Some(x as u32)': Option<u32>
    1230..1231 'x': i32
    1230..1238 'x as u32': u32
    1247..1255 '{ None }': Option<u32>
    1249..1253 'None': Option<u32>
    1273..1283 '|y| { y; }': |u32| -> ()
    1274..1275 'y': u32
    1277..1283 '{ y; }': ()
    1279..1280 'y': u32
    "###
    );
}

#[test]
fn nested_assoc() {
    let t = type_at(
        r#"
//- /main.rs
struct Bar;
struct Foo;

trait A {
    type OutputA;
}

impl A for Bar {
    type OutputA = Foo;
}

trait B {
    type Output;
    fn foo() -> Self::Output;
}

impl<T:A> B for T {
    type Output = T::OutputA;
    fn foo() -> Self::Output { loop {} }
}

fn main() {
    Bar::foo()<|>;
}
"#,
    );
    assert_eq!(t, "Foo");
}

#[test]
fn trait_object_no_coercion() {
    assert_snapshot!(
        infer_with_mismatches(r#"
trait Foo {}

fn foo(x: &dyn Foo) {}

fn test(x: &dyn Foo) {
    foo(x);
}
"#, true),
        @r###"
    22..23 'x': &dyn Foo
    35..37 '{}': ()
    47..48 'x': &dyn Foo
    60..75 '{     foo(x); }': ()
    66..69 'foo': fn foo(&dyn Foo)
    66..72 'foo(x)': ()
    70..71 'x': &dyn Foo
    "###
    );
}

#[test]
fn builtin_copy() {
    assert_snapshot!(
        infer_with_mismatches(r#"
#[lang = "copy"]
trait Copy {}

struct IsCopy;
impl Copy for IsCopy {}
struct NotCopy;

trait Test { fn test(&self) -> bool; }
impl<T: Copy> Test for T {}

fn test() {
    IsCopy.test();
    NotCopy.test();
    (IsCopy, IsCopy).test();
    (IsCopy, NotCopy).test();
}
"#, true),
        @r###"
    111..115 'self': &Self
    167..268 '{     ...t(); }': ()
    173..179 'IsCopy': IsCopy
    173..186 'IsCopy.test()': bool
    192..199 'NotCopy': NotCopy
    192..206 'NotCopy.test()': {unknown}
    212..228 '(IsCop...sCopy)': (IsCopy, IsCopy)
    212..235 '(IsCop...test()': bool
    213..219 'IsCopy': IsCopy
    221..227 'IsCopy': IsCopy
    241..258 '(IsCop...tCopy)': (IsCopy, NotCopy)
    241..265 '(IsCop...test()': {unknown}
    242..248 'IsCopy': IsCopy
    250..257 'NotCopy': NotCopy
    "###
    );
}

#[test]
fn builtin_fn_def_copy() {
    assert_snapshot!(
        infer_with_mismatches(r#"
#[lang = "copy"]
trait Copy {}

fn foo() {}
fn bar<T: Copy>(T) -> T {}
struct Struct(usize);
enum Enum { Variant(usize) }

trait Test { fn test(&self) -> bool; }
impl<T: Copy> Test for T {}

fn test() {
    foo.test();
    bar.test();
    Struct.test();
    Enum::Variant.test();
}
"#, true),
        @r###"
    42..44 '{}': ()
    61..62 'T': {unknown}
    69..71 '{}': ()
    69..71: expected T, got ()
    146..150 'self': &Self
    202..282 '{     ...t(); }': ()
    208..211 'foo': fn foo()
    208..218 'foo.test()': bool
    224..227 'bar': fn bar<{unknown}>({unknown}) -> {unknown}
    224..234 'bar.test()': bool
    240..246 'Struct': Struct(usize) -> Struct
    240..253 'Struct.test()': bool
    259..272 'Enum::Variant': Variant(usize) -> Enum
    259..279 'Enum::...test()': bool
    "###
    );
}

#[test]
fn builtin_fn_ptr_copy() {
    assert_snapshot!(
        infer_with_mismatches(r#"
#[lang = "copy"]
trait Copy {}

trait Test { fn test(&self) -> bool; }
impl<T: Copy> Test for T {}

fn test(f1: fn(), f2: fn(usize) -> u8, f3: fn(u8, u8) -> &u8) {
    f1.test();
    f2.test();
    f3.test();
}
"#, true),
        @r###"
    55..59 'self': &Self
    109..111 'f1': fn()
    119..121 'f2': fn(usize) -> u8
    140..142 'f3': fn(u8, u8) -> &u8
    163..211 '{     ...t(); }': ()
    169..171 'f1': fn()
    169..178 'f1.test()': bool
    184..186 'f2': fn(usize) -> u8
    184..193 'f2.test()': bool
    199..201 'f3': fn(u8, u8) -> &u8
    199..208 'f3.test()': bool
    "###
    );
}

#[test]
fn builtin_sized() {
    assert_snapshot!(
        infer_with_mismatches(r#"
#[lang = "sized"]
trait Sized {}

trait Test { fn test(&self) -> bool; }
impl<T: Sized> Test for T {}

fn test() {
    1u8.test();
    (*"foo").test(); // not Sized
    (1u8, 1u8).test();
    (1u8, *"foo").test(); // not Sized
}
"#, true),
        @r###"
    57..61 'self': &Self
    114..229 '{     ...ized }': ()
    120..123 '1u8': u8
    120..130 '1u8.test()': bool
    136..151 '(*"foo").test()': {unknown}
    137..143 '*"foo"': str
    138..143 '"foo"': &str
    170..180 '(1u8, 1u8)': (u8, u8)
    170..187 '(1u8, ...test()': bool
    171..174 '1u8': u8
    176..179 '1u8': u8
    193..206 '(1u8, *"foo")': (u8, str)
    193..213 '(1u8, ...test()': {unknown}
    194..197 '1u8': u8
    199..205 '*"foo"': str
    200..205 '"foo"': &str
    "###
    );
}

#[test]
fn integer_range_iterate() {
    let t = type_at(
        r#"
//- /main.rs crate:main deps:core
fn test() {
    for x in 0..100 { x<|>; }
}

//- /core.rs crate:core
pub mod ops {
    pub struct Range<Idx> {
        pub start: Idx,
        pub end: Idx,
    }
}

pub mod iter {
    pub trait Iterator {
        type Item;
    }

    pub trait IntoIterator {
        type Item;
        type IntoIter: Iterator<Item = Self::Item>;
    }

    impl<T> IntoIterator for T where T: Iterator {
        type Item = <T as Iterator>::Item;
        type IntoIter = Self;
    }
}

trait Step {}
impl Step for i32 {}
impl Step for i64 {}

impl<A: Step> iter::Iterator for ops::Range<A> {
    type Item = A;
}
"#,
    );
    assert_eq!(t, "i32");
}

#[test]
fn infer_closure_arg() {
    assert_snapshot!(
        infer(
            r#"
            //- /lib.rs

            enum Option<T> {
                None,
                Some(T)
            }

            fn foo() {
                let s = Option::None;
                let f = |x: Option<i32>| {};
                (&f)(s)
            }
        "#
        ),
        @r###"
    137..259 '{     ...     }': ()
    159..160 's': Option<i32>
    163..175 'Option::None': Option<i32>
    197..198 'f': |Option<i32>| -> ()
    201..220 '|x: Op...2>| {}': |Option<i32>| -> ()
    202..203 'x': Option<i32>
    218..220 '{}': ()
    238..245 '(&f)(s)': ()
    239..241 '&f': &|Option<i32>| -> ()
    240..241 'f': |Option<i32>| -> ()
    243..244 's': Option<i32>
    "###
    );
}

#[test]
fn infer_fn_trait_arg() {
    assert_snapshot!(
        infer(
            r#"
            //- /lib.rs deps:std

            #[lang = "fn_once"]
            pub trait FnOnce<Args> {
                type Output;

                extern "rust-call" fn call_once(&self, args: Args) -> Self::Output;
            }

            #[lang = "fn"]
            pub trait Fn<Args>:FnOnce<Args> {
                extern "rust-call" fn call(&self, args: Args) -> Self::Output;
            }

            enum Option<T> {
                None,
                Some(T)
            }

            fn foo<F, T>(f: F) -> T
            where
                F: Fn(Option<i32>) -> T,
            {
                let s = None;
                f(s)
            }
        "#
        ),
        @r###"
    183..187 'self': &Self
    189..193 'args': Args
    350..354 'self': &Self
    356..360 'args': Args
    515..516 'f': F
    597..663 '{     ...     }': T
    619..620 's': Option<i32>
    623..627 'None': Option<i32>
    645..646 'f': F
    645..649 'f(s)': T
    647..648 's': Option<i32>
    "###
    );
}

#[test]
fn infer_box_fn_arg() {
    assert_snapshot!(
        infer(
            r#"
            //- /lib.rs deps:std

            #[lang = "fn_once"]
            pub trait FnOnce<Args> {
                type Output;

                extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
            }

            #[lang = "deref"]
            pub trait Deref {
                type Target: ?Sized;

                fn deref(&self) -> &Self::Target;
            }

            #[lang = "owned_box"]
            pub struct Box<T: ?Sized> {
                inner: *mut T,
            }

            impl<T: ?Sized> Deref for Box<T> {
                type Target = T;

                fn deref(&self) -> &T {
                    &self.inner
                }
            }

            enum Option<T> {
                None,
                Some(T)
            }

            fn foo() {
                let s = Option::None;
                let f: Box<dyn FnOnce(&Option<i32>)> = box (|ps| {});
                f(&s)
            }
        "#
        ),
        @r###"
    182..186 'self': Self
    188..192 'args': Args
    356..360 'self': &Self
    622..626 'self': &Box<T>
    634..685 '{     ...     }': &T
    656..667 '&self.inner': &*mut T
    657..661 'self': &Box<T>
    657..667 'self.inner': *mut T
    812..957 '{     ...     }': FnOnce::Output<dyn FnOnce<(&Option<i32>,)>, (&Option<i32>,)>
    834..835 's': Option<i32>
    838..850 'Option::None': Option<i32>
    872..873 'f': Box<dyn FnOnce<(&Option<i32>,)>>
    907..920 'box (|ps| {})': Box<|{unknown}| -> ()>
    912..919 '|ps| {}': |{unknown}| -> ()
    913..915 'ps': {unknown}
    917..919 '{}': ()
    938..939 'f': Box<dyn FnOnce<(&Option<i32>,)>>
    938..943 'f(&s)': FnOnce::Output<dyn FnOnce<(&Option<i32>,)>, (&Option<i32>,)>
    940..942 '&s': &Option<i32>
    941..942 's': Option<i32>
    "###
    );
}

#[test]
fn infer_dyn_fn_output() {
    assert_snapshot!(
        infer(
            r#"
            //- /lib.rs deps:std

            #[lang = "fn_once"]
            pub trait FnOnce<Args> {
                type Output;

                extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
            }

            #[lang = "fn"]
            pub trait Fn<Args>:FnOnce<Args> {
                extern "rust-call" fn call(&self, args: Args) -> Self::Output;
            }

            #[lang = "deref"]
            pub trait Deref {
                type Target: ?Sized;

                fn deref(&self) -> &Self::Target;
            }

            #[lang = "owned_box"]
            pub struct Box<T: ?Sized> {
                inner: *mut T,
            }

            impl<T: ?Sized> Deref for Box<T> {
                type Target = T;

                fn deref(&self) -> &T {
                    &self.inner
                }
            }

            fn foo() {
                let f: Box<dyn Fn() -> i32> = box(|| 5);
                let x = f();
            }
        "#
        ),
        @r###"
    182..186 'self': Self
    188..192 'args': Args
    349..353 'self': &Self
    355..359 'args': Args
    523..527 'self': &Self
    789..793 'self': &Box<T>
    801..852 '{     ...     }': &T
    823..834 '&self.inner': &*mut T
    824..828 'self': &Box<T>
    824..834 'self.inner': *mut T
    889..990 '{     ...     }': ()
    911..912 'f': Box<dyn Fn<(), Output = i32>>
    937..946 'box(|| 5)': Box<|| -> i32>
    941..945 '|| 5': || -> i32
    944..945 '5': i32
    968..969 'x': FnOnce::Output<dyn Fn<(), Output = i32>, ()>
    972..973 'f': Box<dyn Fn<(), Output = i32>>
    972..975 'f()': FnOnce::Output<dyn Fn<(), Output = i32>, ()>
    "###
    );
}
