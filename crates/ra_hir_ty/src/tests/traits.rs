use insta::assert_snapshot;

use ra_db::fixture::WithFixture;
use test_utils::covers;

use super::{infer, infer_with_mismatches, type_at, type_at_pos};
use crate::test_db::TestDB;

#[test]
fn infer_await() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std

struct IntFuture;

impl Future for IntFuture {
    type Output = u64;
}

fn test() {
    let r = IntFuture;
    let v = r.await;
    v<|>;
}

//- /std.rs crate:std
#[prelude_import] use future::*;
mod future {
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
//- /main.rs crate:main deps:std

async fn foo() -> u64 {
    128
}

fn test() {
    let r = foo();
    let v = r.await;
    v<|>;
}

//- /std.rs crate:std
#[prelude_import] use future::*;
mod future {
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
//- /main.rs crate:main deps:std

async fn foo() -> u64 {
    128
}

fn test() {
    let r = foo();
    r<|>;
}

//- /std.rs crate:std
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
//- /main.rs crate:main deps:std

fn test() {
    let r: Result<i32, u64> = Result::Ok(1);
    let v = r?;
    v<|>;
}

//- /std.rs crate:std

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
//- /main.rs crate:main deps:std

use std::collections::Vec;

fn test() {
    let v = Vec::new();
    v.push("foo");
    for x in v {
        x<|>;
    }
}

//- /std.rs crate:std

#[prelude_import] use iter::*;
mod iter {
    trait IntoIterator {
        type Item;
    }
}

mod collections {
    struct Vec<T> {}
    impl<T> Vec<T> {
        fn new() -> Self { Vec {} }
        fn push(&mut self, t: T) { }
    }

    impl<T> crate::iter::IntoIterator for Vec<T> {
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
    [86; 87) 't': T
    [92; 94) '{}': ()
    [105; 144) '{     ...(s); }': ()
    [115; 116) 's': S<u32>
    [119; 120) 'S': S<u32>(T) -> S<T>
    [119; 129) 'S(unknown)': S<u32>
    [121; 128) 'unknown': u32
    [135; 138) 'foo': fn foo<S<u32>>(T) -> ()
    [135; 141) 'foo(s)': ()
    [139; 140) 's': S<u32>
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
    [87; 88) 't': T
    [98; 100) '{}': ()
    [111; 163) '{     ...(s); }': ()
    [121; 122) 's': S<u32>
    [125; 126) 'S': S<u32>(T) -> S<T>
    [125; 135) 'S(unknown)': S<u32>
    [127; 134) 'unknown': u32
    [145; 146) 'x': u32
    [154; 157) 'foo': fn foo<u32, S<u32>>(T) -> U
    [154; 160) 'foo(s)': u32
    [158; 159) 's': S<u32>
    "###
    );
}

#[test]
fn infer_project_associated_type() {
    // y, z, a don't yet work because of https://github.com/rust-lang/chalk/issues/234
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
    [108; 261) '{     ...ter; }': ()
    [118; 119) 'x': u32
    [145; 146) '1': u32
    [156; 157) 'y': {unknown}
    [183; 192) 'no_matter': {unknown}
    [202; 203) 'z': {unknown}
    [215; 224) 'no_matter': {unknown}
    [234; 235) 'a': {unknown}
    [249; 258) 'no_matter': {unknown}
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
    [106; 107) 't': T
    [123; 125) '{}': ()
    [147; 148) 't': T
    [178; 180) '{}': ()
    [202; 203) 't': T
    [221; 223) '{}': ()
    [234; 300) '{     ...(S); }': ()
    [244; 245) 'x': u32
    [248; 252) 'foo1': fn foo1<S>(T) -> <T as Iterable>::Item
    [248; 255) 'foo1(S)': u32
    [253; 254) 'S': S
    [265; 266) 'y': u32
    [269; 273) 'foo2': fn foo2<S>(T) -> <T as Iterable>::Item
    [269; 276) 'foo2(S)': u32
    [274; 275) 'S': S
    [286; 287) 'z': u32
    [290; 294) 'foo3': fn foo3<S>(T) -> <T as Iterable>::Item
    [290; 297) 'foo3(S)': u32
    [295; 296) 'S': S
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
    [67; 100) '{     ...own; }': ()
    [77; 78) 'y': {unknown}
    [90; 97) 'unknown': {unknown}
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
    [16; 17) '1': u32
    [16; 21) '1 + 1': u32
    [20; 21) '1': u32
    [39; 55) '{ let ...1; x }': u64
    [45; 46) 'x': u64
    [49; 50) '1': u64
    [52; 53) 'x': u64
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
    [38; 87) '{     ... a.1 }': u64
    [48; 49) 'a': S
    [52; 53) 'S': S(i32, u64) -> S
    [52; 59) 'S(4, 6)': S
    [54; 55) '4': i32
    [57; 58) '6': u64
    [69; 70) 'b': i32
    [73; 74) 'a': S
    [73; 76) 'a.0': i32
    [82; 83) 'a': S
    [82; 85) 'a.1': u64
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
    [44; 102) '{     ...0(2) }': u64
    [54; 55) 'a': S
    [58; 59) 'S': S(fn(u32) -> u64) -> S
    [58; 68) 'S(|i| 2*i)': S
    [60; 67) '|i| 2*i': |u32| -> u64
    [61; 62) 'i': u32
    [64; 65) '2': u32
    [64; 67) '2*i': u32
    [66; 67) 'i': u32
    [78; 79) 'b': u64
    [82; 83) 'a': S
    [82; 85) 'a.0': fn(u32) -> u64
    [82; 88) 'a.0(4)': u64
    [86; 87) '4': u32
    [94; 95) 'a': S
    [94; 97) 'a.0': fn(u32) -> u64
    [94; 100) 'a.0(2)': u64
    [98; 99) '2': u32
    "###
    );
}

#[test]
fn indexing_arrays() {
    assert_snapshot!(
        infer("fn main() { &mut [9][2]; }"),
        @r###"
    [10; 26) '{ &mut...[2]; }': ()
    [12; 23) '&mut [9][2]': &mut {unknown}
    [17; 20) '[9]': [i32;_]
    [17; 23) '[9][2]': {unknown}
    [18; 19) '9': i32
    [21; 22) '2': i32
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
    let b = a[1];
    b<|>;
}

//- /std.rs crate:std

#[prelude_import] use ops::*;
mod ops {
    pub trait Index<Idx> {
        type Output;
    }
}
"#,
    );
    assert_eq!("Foo", type_at_pos(&db, pos));
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
    // FIXME: fix type parameter names going missing when going through Chalk
    assert_eq!(t, "ApplyL::Out<[missing name]>");
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
    // FIXME here Chalk doesn't normalize the type to a placeholder. I think we
    // need to add a rule like Normalize(<T as ApplyL>::Out -> ApplyL::Out<T>)
    // to the trait env ourselves here; probably Chalk can't do this by itself.
    // assert_eq!(t, "ApplyL::Out<[missing name]>");
    assert_eq!(t, "{unknown}");
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
    [30; 34) 'self': &Self
    [55; 59) 'self': &Self
    [99; 101) '{}': ()
    [111; 112) 'x': impl Trait<u64>
    [131; 132) 'y': &impl Trait<u64>
    [152; 269) '{     ...2(); }': ()
    [158; 159) 'x': impl Trait<u64>
    [165; 166) 'y': &impl Trait<u64>
    [176; 177) 'z': impl Trait<u64>
    [180; 183) 'bar': fn bar() -> impl Trait<u64>
    [180; 185) 'bar()': impl Trait<u64>
    [191; 192) 'x': impl Trait<u64>
    [191; 198) 'x.foo()': u64
    [204; 205) 'y': &impl Trait<u64>
    [204; 211) 'y.foo()': u64
    [217; 218) 'z': impl Trait<u64>
    [217; 224) 'z.foo()': u64
    [230; 231) 'x': impl Trait<u64>
    [230; 238) 'x.foo2()': i64
    [244; 245) 'y': &impl Trait<u64>
    [244; 252) 'y.foo2()': i64
    [258; 259) 'z': impl Trait<u64>
    [258; 266) 'z.foo2()': i64
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
    [30; 34) 'self': &Self
    [55; 59) 'self': &Self
    [98; 100) '{}': ()
    [110; 111) 'x': dyn Trait<u64>
    [129; 130) 'y': &dyn Trait<u64>
    [149; 266) '{     ...2(); }': ()
    [155; 156) 'x': dyn Trait<u64>
    [162; 163) 'y': &dyn Trait<u64>
    [173; 174) 'z': dyn Trait<u64>
    [177; 180) 'bar': fn bar() -> dyn Trait<u64>
    [177; 182) 'bar()': dyn Trait<u64>
    [188; 189) 'x': dyn Trait<u64>
    [188; 195) 'x.foo()': u64
    [201; 202) 'y': &dyn Trait<u64>
    [201; 208) 'y.foo()': u64
    [214; 215) 'z': dyn Trait<u64>
    [214; 221) 'z.foo()': u64
    [227; 228) 'x': dyn Trait<u64>
    [227; 235) 'x.foo2()': i64
    [241; 242) 'y': &dyn Trait<u64>
    [241; 249) 'y.foo2()': i64
    [255; 256) 'z': dyn Trait<u64>
    [255; 263) 'z.foo2()': i64
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
    [27; 31) 'self': &Self
    [61; 63) '{}': ()
    [73; 74) 'x': dyn Trait
    [83; 84) 'y': &dyn Trait
    [101; 176) '{     ...o(); }': ()
    [107; 108) 'x': dyn Trait
    [114; 115) 'y': &dyn Trait
    [125; 126) 'z': dyn Trait
    [129; 132) 'bar': fn bar() -> dyn Trait
    [129; 134) 'bar()': dyn Trait
    [140; 141) 'x': dyn Trait
    [140; 147) 'x.foo()': u64
    [153; 154) 'y': &dyn Trait
    [153; 160) 'y.foo()': u64
    [166; 167) 'z': dyn Trait
    [166; 173) 'z.foo()': u64
    "###
    );
}

#[test]
fn weird_bounds() {
    assert_snapshot!(
        infer(r#"
trait Trait {}
fn test() {
    let a: impl Trait + 'lifetime = foo;
    let b: impl 'lifetime = foo;
    let b: impl (Trait) = foo;
    let b: impl ('lifetime) = foo;
    let d: impl ?Sized = foo;
    let e: impl Trait + ?Sized = foo;
}
"#),
        @r###"
    [26; 237) '{     ...foo; }': ()
    [36; 37) 'a': impl Trait + {error}
    [64; 67) 'foo': impl Trait + {error}
    [77; 78) 'b': impl {error}
    [97; 100) 'foo': impl {error}
    [110; 111) 'b': impl Trait
    [128; 131) 'foo': impl Trait
    [141; 142) 'b': impl {error}
    [163; 166) 'foo': impl {error}
    [176; 177) 'd': impl {error}
    [193; 196) 'foo': impl {error}
    [206; 207) 'e': impl Trait + {error}
    [231; 234) 'foo': impl Trait + {error}
    "###
    );
}

#[test]
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
    [50; 51) 't': T
    [78; 80) '{}': ()
    [112; 113) 't': T
    [123; 125) '{}': ()
    [155; 156) 't': T
    [166; 169) '{t}': T
    [167; 168) 't': T
    [257; 258) 'x': T
    [263; 264) 'y': impl Trait<Type = i64>
    [290; 398) '{     ...r>); }': ()
    [296; 299) 'get': fn get<T>(T) -> <T as Trait>::Type
    [296; 302) 'get(x)': {unknown}
    [300; 301) 'x': T
    [308; 312) 'get2': fn get2<{unknown}, T>(T) -> U
    [308; 315) 'get2(x)': {unknown}
    [313; 314) 'x': T
    [321; 324) 'get': fn get<impl Trait<Type = i64>>(T) -> <T as Trait>::Type
    [321; 327) 'get(y)': {unknown}
    [325; 326) 'y': impl Trait<Type = i64>
    [333; 337) 'get2': fn get2<{unknown}, impl Trait<Type = i64>>(T) -> U
    [333; 340) 'get2(y)': {unknown}
    [338; 339) 'y': impl Trait<Type = i64>
    [346; 349) 'get': fn get<S<u64>>(T) -> <T as Trait>::Type
    [346; 357) 'get(set(S))': u64
    [350; 353) 'set': fn set<S<u64>>(T) -> T
    [350; 356) 'set(S)': S<u64>
    [354; 355) 'S': S<u64>
    [363; 367) 'get2': fn get2<u64, S<u64>>(T) -> U
    [363; 375) 'get2(set(S))': u64
    [368; 371) 'set': fn set<S<u64>>(T) -> T
    [368; 374) 'set(S)': S<u64>
    [372; 373) 'S': S<u64>
    [381; 385) 'get2': fn get2<str, S<str>>(T) -> U
    [381; 395) 'get2(S::<str>)': str
    [386; 394) 'S::<str>': S<str>
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
    // std::env::set_var("CHALK_DEBUG", "1");
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
    [62; 66) 'self': Self
    [164; 165) 'x': T
    [170; 186) '{     ...o(); }': ()
    [176; 177) 'x': T
    [176; 183) 'x.foo()': {unknown}
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
    [50; 54) 'self': &Self
    [63; 65) '{}': ()
    [182; 183) 'x': T
    [188; 189) 'y': U
    [194; 223) '{     ...o(); }': ()
    [200; 201) 'x': T
    [200; 207) 'x.foo()': u32
    [213; 214) 'y': U
    [213; 220) 'y.foo()': u32
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
    [44; 45) 'x': T
    [50; 66) '{     ...o(); }': ()
    [56; 57) 'x': T
    [56; 63) 'x.foo()': {unknown}
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
    [103; 104) 't': T
    [114; 116) '{}': ()
    [146; 147) 't': T
    [157; 160) '{t}': T
    [158; 159) 't': T
    [259; 280) '{     ...S)); }': ()
    [265; 269) 'get2': fn get2<u64, S<u64>>(T) -> U
    [265; 277) 'get2(set(S))': u64
    [270; 273) 'set': fn set<S<u64>>(T) -> T
    [270; 276) 'set(S)': S<u64>
    [274; 275) 'S': S<u64>
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
    [57; 61) 'self': Self
    [63; 67) 'args': Args
    [150; 151) 'f': F
    [156; 184) '{     ...2)); }': ()
    [162; 163) 'f': F
    [162; 181) 'f.call...1, 2))': {unknown}
    [174; 180) '(1, 2)': (u32, u64)
    [175; 176) '1': u32
    [178; 179) '2': u64
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
    [148; 152) 'self': Option<T>
    [154; 155) 'f': F
    [173; 175) '{}': ()
    [189; 308) '{     ... 1); }': ()
    [199; 200) 'x': Option<u32>
    [203; 215) 'Option::Some': Some<u32>(T) -> Option<T>
    [203; 221) 'Option...(1u32)': Option<u32>
    [216; 220) '1u32': u32
    [227; 228) 'x': Option<u32>
    [227; 243) 'x.map(...v + 1)': Option<u32>
    [233; 242) '|v| v + 1': |u32| -> u32
    [234; 235) 'v': u32
    [237; 238) 'v': u32
    [237; 242) 'v + 1': u32
    [241; 242) '1': u32
    [249; 250) 'x': Option<u32>
    [249; 265) 'x.map(... 1u64)': Option<u64>
    [255; 264) '|_v| 1u64': |u32| -> u64
    [256; 258) '_v': u32
    [260; 264) '1u64': u64
    [275; 276) 'y': Option<i64>
    [292; 293) 'x': Option<u32>
    [292; 305) 'x.map(|_v| 1)': Option<i64>
    [298; 304) '|_v| 1': |u32| -> i64
    [299; 301) '_v': u32
    [303; 304) '1': i64
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
    [73; 74) 'f': F
    [79; 155) '{     ...+ v; }': ()
    [85; 86) 'f': F
    [85; 89) 'f(1)': {unknown}
    [87; 88) '1': i32
    [99; 100) 'g': |u64| -> i32
    [103; 112) '|v| v + 1': |u64| -> i32
    [104; 105) 'v': u64
    [107; 108) 'v': u64
    [107; 112) 'v + 1': i32
    [111; 112) '1': i32
    [118; 119) 'g': |u64| -> i32
    [118; 125) 'g(1u64)': i32
    [120; 124) '1u64': u64
    [135; 136) 'h': |u128| -> u128
    [139; 152) '|v| 1u128 + v': |u128| -> u128
    [140; 141) 'v': u128
    [143; 148) '1u128': u128
    [143; 152) '1u128 + v': u128
    [151; 152) 'v': u128
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
    [95; 96) 'x': T
    [101; 102) 'f': F
    [112; 114) '{}': ()
    [148; 149) 'f': F
    [154; 155) 'x': T
    [165; 167) '{}': ()
    [202; 206) 'self': S
    [254; 258) 'self': S
    [260; 261) 'x': T
    [266; 267) 'f': F
    [277; 279) '{}': ()
    [317; 321) 'self': S
    [323; 324) 'f': F
    [329; 330) 'x': T
    [340; 342) '{}': ()
    [356; 515) '{     ... S); }': ()
    [366; 368) 'x1': u64
    [371; 375) 'foo1': fn foo1<S, u64, |S| -> u64>(T, F) -> U
    [371; 394) 'foo1(S...hod())': u64
    [376; 377) 'S': S
    [379; 393) '|s| s.method()': |S| -> u64
    [380; 381) 's': S
    [383; 384) 's': S
    [383; 393) 's.method()': u64
    [404; 406) 'x2': u64
    [409; 413) 'foo2': fn foo2<S, u64, |S| -> u64>(F, T) -> U
    [409; 432) 'foo2(|...(), S)': u64
    [414; 428) '|s| s.method()': |S| -> u64
    [415; 416) 's': S
    [418; 419) 's': S
    [418; 428) 's.method()': u64
    [430; 431) 'S': S
    [442; 444) 'x3': u64
    [447; 448) 'S': S
    [447; 472) 'S.foo1...hod())': u64
    [454; 455) 'S': S
    [457; 471) '|s| s.method()': |S| -> u64
    [458; 459) 's': S
    [461; 462) 's': S
    [461; 471) 's.method()': u64
    [482; 484) 'x4': u64
    [487; 488) 'S': S
    [487; 512) 'S.foo2...(), S)': u64
    [494; 508) '|s| s.method()': |S| -> u64
    [495; 496) 's': S
    [498; 499) 's': S
    [498; 508) 's.method()': u64
    [510; 511) 'S': S
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
fn unify_impl_trait() {
    covers!(insert_vars_for_impl_trait);
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
    [27; 28) 'x': impl Trait<u32>
    [47; 58) '{ loop {} }': ()
    [49; 56) 'loop {}': !
    [54; 56) '{}': ()
    [69; 70) 'x': impl Trait<T>
    [92; 103) '{ loop {} }': T
    [94; 101) 'loop {}': !
    [99; 101) '{}': ()
    [172; 183) '{ loop {} }': T
    [174; 181) 'loop {}': !
    [179; 181) '{}': ()
    [214; 310) '{     ...t()) }': S<i32>
    [224; 226) 's1': S<u32>
    [229; 230) 'S': S<u32>(T) -> S<T>
    [229; 241) 'S(default())': S<u32>
    [231; 238) 'default': fn default<u32>() -> T
    [231; 240) 'default()': u32
    [247; 250) 'foo': fn foo(impl Trait<u32>) -> ()
    [247; 254) 'foo(s1)': ()
    [251; 253) 's1': S<u32>
    [264; 265) 'x': i32
    [273; 276) 'bar': fn bar<i32>(impl Trait<T>) -> T
    [273; 290) 'bar(S(...lt()))': i32
    [277; 278) 'S': S<i32>(T) -> S<T>
    [277; 289) 'S(default())': S<i32>
    [279; 286) 'default': fn default<i32>() -> T
    [279; 288) 'default()': i32
    [296; 297) 'S': S<i32>(T) -> S<T>
    [296; 308) 'S(default())': S<i32>
    [298; 305) 'default': fn default<i32>() -> T
    [298; 307) 'default()': i32
    "###
    );
}
