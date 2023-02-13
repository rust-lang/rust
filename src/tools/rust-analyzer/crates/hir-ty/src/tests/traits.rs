use cov_mark::check;
use expect_test::expect;

use super::{check, check_infer, check_infer_with_mismatches, check_no_mismatches, check_types};

#[test]
fn infer_await() {
    check_types(
        r#"
//- minicore: future
struct IntFuture;

impl core::future::Future for IntFuture {
    type Output = u64;
}

fn test() {
    let r = IntFuture;
    let v = r.await;
    v;
} //^ u64
"#,
    );
}

#[test]
fn infer_async() {
    check_types(
        r#"
//- minicore: future
async fn foo() -> u64 { 128 }

fn test() {
    let r = foo();
    let v = r.await;
    v;
} //^ u64
"#,
    );
}

#[test]
fn infer_desugar_async() {
    check_types(
        r#"
//- minicore: future, sized
async fn foo() -> u64 { 128 }

fn test() {
    let r = foo();
    r;
} //^ impl Future<Output = u64>
"#,
    );
}

#[test]
fn infer_async_block() {
    check_types(
        r#"
//- minicore: future, option
async fn test() {
    let a = async { 42 };
    a;
//  ^ impl Future<Output = i32>
    let x = a.await;
    x;
//  ^ i32
    let b = async {}.await;
    b;
//  ^ ()
    let c = async {
        let y = None;
        y
    //  ^ Option<u64>
    };
    let _: Option<u64> = c.await;
    c;
//  ^ impl Future<Output = Option<u64>>
}
"#,
    );
}

#[test]
fn auto_sized_async_block() {
    check_no_mismatches(
        r#"
//- minicore: future, sized

use core::future::Future;
struct MyFut<Fut>(Fut);

impl<Fut> Future for MyFut<Fut>
where Fut: Future
{
    type Output = Fut::Output;
}
async fn reproduction() -> usize {
    let f = async {999usize};
    MyFut(f).await
}
    "#,
    );
    check_no_mismatches(
        r#"
//- minicore: future
//#11815
#[lang = "sized"]
pub trait Sized {}

#[lang = "unsize"]
pub trait Unsize<T: ?Sized> {}

#[lang = "coerce_unsized"]
pub trait CoerceUnsized<T> {}

pub unsafe trait Allocator {}

pub struct Global;
unsafe impl Allocator for Global {}

#[lang = "owned_box"]
#[fundamental]
pub struct Box<T: ?Sized, A: Allocator = Global>;

impl<T: ?Sized + Unsize<U>, U: ?Sized, A: Allocator> CoerceUnsized<Box<U, A>> for Box<T, A> {}

fn send() ->  Box<dyn Future<Output = ()> + Send + 'static>{
    box async move {}
}

fn not_send() -> Box<dyn Future<Output = ()> + 'static> {
    box async move {}
}
    "#,
    );
}

#[test]
fn into_future_trait() {
    check_types(
        r#"
//- minicore: future
struct Futurable;
impl core::future::IntoFuture for Futurable {
    type Output = u64;
    type IntoFuture = IntFuture;
}

struct IntFuture;
impl core::future::Future for IntFuture {
    type Output = u64;
}

fn test() {
    let r = Futurable;
    let v = r.await;
    v;
} //^ u64
"#,
    );
}

#[test]
fn infer_try_trait() {
    check_types(
        r#"
//- minicore: try, result
fn test() {
    let r: Result<i32, u64> = Result::Ok(1);
    let v = r?;
    v;
} //^ i32

impl<O, E> core::ops::Try for Result<O, E> {
    type Output = O;
    type Error = Result<core::convert::Infallible, E>;
}

impl<T, E, F: From<E>> core::ops::FromResidual<Result<core::convert::Infallible, E>> for Result<T, F> {}
"#,
    );
}

#[test]
fn infer_for_loop() {
    check_types(
        r#"
//- minicore: iterator
//- /main.rs crate:main deps:alloc
#![no_std]
use alloc::collections::Vec;

fn test() {
    let v = Vec::new();
    v.push("foo");
    for x in v {
        x;
    } //^ &str
}

//- /alloc.rs crate:alloc
#![no_std]
pub mod collections {
    pub struct Vec<T> {}
    impl<T> Vec<T> {
        pub fn new() -> Self { Vec {} }
        pub fn push(&mut self, t: T) { }
    }

    impl<T> IntoIterator for Vec<T> {
        type Item = T;
        type IntoIter = IntoIter<T>;
    }

    struct IntoIter<T> {}
    impl<T> Iterator for IntoIter<T> {
        type Item = T;
    }
}
"#,
    );
}

#[test]
fn infer_ops_neg() {
    check_types(
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
    b;
} //^ Foo

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
}

#[test]
fn infer_ops_not() {
    check_types(
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
    b;
} //^ Foo

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
}

#[test]
fn infer_from_bound_1() {
    check_types(
        r#"
trait Trait<T> {}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn foo<T: Trait<u32>>(t: T) {}
fn test() {
    let s = S(unknown);
           // ^^^^^^^ u32
    foo(s);
}"#,
    );
}

#[test]
fn infer_from_bound_2() {
    check_types(
        r#"
trait Trait<T> {}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn foo<U, T: Trait<U>>(t: T) -> U { loop {} }
fn test() {
    let s = S(unknown);
           // ^^^^^^^ u32
    let x: u32 = foo(s);
}"#,
    );
}

#[test]
fn trait_default_method_self_bound_implements_trait() {
    cov_mark::check!(trait_self_implements_self);
    check(
        r#"
trait Trait {
    fn foo(&self) -> i64;
    fn bar(&self) -> () {
        self.foo();
     // ^^^^^^^^^^ type: i64
    }
}"#,
    );
}

#[test]
fn trait_default_method_self_bound_implements_super_trait() {
    check(
        r#"
trait SuperTrait {
    fn foo(&self) -> i64;
}
trait Trait: SuperTrait {
    fn bar(&self) -> () {
        self.foo();
     // ^^^^^^^^^^ type: i64
    }
}"#,
    );
}

#[test]
fn infer_project_associated_type() {
    check_types(
        r#"
trait Iterable {
    type Item;
}
struct S;
impl Iterable for S { type Item = u32; }
fn test<T: Iterable>() {
    let x: <S as Iterable>::Item = 1;
                                // ^ u32
    let y: <T as Iterable>::Item = u;
                                // ^ Iterable::Item<T>
    let z: T::Item = u;
                  // ^ Iterable::Item<T>
    let a: <T>::Item = u;
                    // ^ Iterable::Item<T>
}"#,
    );
}

#[test]
fn infer_return_associated_type() {
    check_types(
        r#"
trait Iterable {
    type Item;
}
struct S;
impl Iterable for S { type Item = u32; }
fn foo1<T: Iterable>(t: T) -> T::Item { loop {} }
fn foo2<T: Iterable>(t: T) -> <T as Iterable>::Item { loop {} }
fn foo3<T: Iterable>(t: T) -> <T>::Item { loop {} }
fn test() {
    foo1(S);
 // ^^^^^^^ u32
    foo2(S);
 // ^^^^^^^ u32
    foo3(S);
 // ^^^^^^^ u32
}"#,
    );
}

#[test]
fn associated_type_shorthand_from_method_bound() {
    check_types(
        r#"
trait Iterable {
    type Item;
}
struct S<T>;
impl<T> S<T> {
    fn foo(self) -> T::Item where T: Iterable { loop {} }
}
fn test<T: Iterable>() {
    let s: S<T>;
    s.foo();
 // ^^^^^^^ Iterable::Item<T>
}"#,
    );
}

#[test]
fn associated_type_shorthand_from_self_issue_12484() {
    check_types(
        r#"
trait Bar {
    type A;
}
trait Foo {
    type A;
    fn test(a: Self::A, _: impl Bar) {
        a;
      //^ Foo::A<Self>
    }
}"#,
    );
}

#[test]
fn infer_associated_type_bound() {
    check_types(
        r#"
trait Iterable {
    type Item;
}
fn test<T: Iterable<Item=u32>>() {
    let y: T::Item = unknown;
                  // ^^^^^^^ u32
}"#,
    );
}

#[test]
fn infer_const_body() {
    // FIXME make check_types work with other bodies
    check_infer(
        r#"
const A: u32 = 1 + 1;
static B: u64 = { let x = 1; x };
"#,
        expect![[r#"
            15..16 '1': u32
            15..20 '1 + 1': u32
            19..20 '1': u32
            38..54 '{ let ...1; x }': u64
            44..45 'x': u64
            48..49 '1': u64
            51..52 'x': u64
        "#]],
    );
}

#[test]
fn tuple_struct_fields() {
    check_infer(
        r#"
struct S(i32, u64);
fn test() -> u64 {
    let a = S(4, 6);
    let b = a.0;
    a.1
}"#,
        expect![[r#"
            37..86 '{     ... a.1 }': u64
            47..48 'a': S
            51..52 'S': S(i32, u64) -> S
            51..58 'S(4, 6)': S
            53..54 '4': i32
            56..57 '6': u64
            68..69 'b': i32
            72..73 'a': S
            72..75 'a.0': i32
            81..82 'a': S
            81..84 'a.1': u64
        "#]],
    );
}

#[test]
fn tuple_struct_with_fn() {
    check_infer(
        r#"
struct S(fn(u32) -> u64);
fn test() -> u64 {
    let a = S(|i| 2*i);
    let b = a.0(4);
    a.0(2)
}"#,
        expect![[r#"
            43..101 '{     ...0(2) }': u64
            53..54 'a': S
            57..58 'S': S(fn(u32) -> u64) -> S
            57..67 'S(|i| 2*i)': S
            59..66 '|i| 2*i': |u32| -> u64
            60..61 'i': u32
            63..64 '2': u32
            63..66 '2*i': u32
            65..66 'i': u32
            77..78 'b': u64
            81..82 'a': S
            81..84 'a.0': fn(u32) -> u64
            81..87 'a.0(4)': u64
            85..86 '4': u32
            93..94 'a': S
            93..96 'a.0': fn(u32) -> u64
            93..99 'a.0(2)': u64
            97..98 '2': u32
        "#]],
    );
}

#[test]
fn indexing_arrays() {
    check_infer(
        "fn main() { &mut [9][2]; }",
        expect![[r#"
            10..26 '{ &mut...[2]; }': ()
            12..23 '&mut [9][2]': &mut {unknown}
            17..20 '[9]': [i32; 1]
            17..23 '[9][2]': {unknown}
            18..19 '9': i32
            21..22 '2': i32
        "#]],
    )
}

#[test]
fn infer_ops_index() {
    check_types(
        r#"
//- minicore: index
struct Bar;
struct Foo;

impl core::ops::Index<u32> for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = a[1u32];
    b;
} //^ Foo
"#,
    );
}

#[test]
fn infer_ops_index_field() {
    check_types(
        r#"
//- minicore: index
struct Bar;
struct Foo {
    field: u32;
}

impl core::ops::Index<u32> for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = a[1u32].field;
    b;
} //^ u32
"#,
    );
}

#[test]
fn infer_ops_index_field_autoderef() {
    check_types(
        r#"
//- minicore: index
struct Bar;
struct Foo {
    field: u32;
}

impl core::ops::Index<u32> for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = (&a[1u32]).field;
    b;
} //^ u32
"#,
    );
}

#[test]
fn infer_ops_index_int() {
    check_types(
        r#"
//- minicore: index
struct Bar;
struct Foo;

impl core::ops::Index<u32> for Bar {
    type Output = Foo;
}

struct Range;
impl core::ops::Index<Range> for Bar {
    type Output = Bar;
}

fn test() {
    let a = Bar;
    let b = a[1];
    b;
  //^ Foo
}
"#,
    );
}

#[test]
fn infer_ops_index_autoderef() {
    check_types(
        r#"
//- minicore: index, slice
fn test() {
    let a = &[1u32, 2, 3];
    let b = a[1];
    b;
} //^ u32
"#,
    );
}

#[test]
fn deref_trait() {
    check_types(
        r#"
//- minicore: deref
struct Arc<T: ?Sized>;
impl<T: ?Sized> core::ops::Deref for Arc<T> {
    type Target = T;
}

struct S;
impl S {
    fn foo(&self) -> u128 { 0 }
}

fn test(s: Arc<S>) {
    (*s, s.foo());
} //^^^^^^^^^^^^^ (S, u128)
"#,
    );
}

#[test]
fn deref_trait_with_inference_var() {
    check_types(
        r#"
//- minicore: deref
struct Arc<T: ?Sized>;
fn new_arc<T: ?Sized>() -> Arc<T> { Arc }
impl<T: ?Sized> core::ops::Deref for Arc<T> {
    type Target = T;
}

struct S;
fn foo(a: Arc<S>) {}

fn test() {
    let a = new_arc();
    let b = *a;
          //^^ S
    foo(a);
}
"#,
    );
}

#[test]
fn deref_trait_infinite_recursion() {
    check_types(
        r#"
//- minicore: deref
struct S;

impl core::ops::Deref for S {
    type Target = S;
}

fn test(s: S) {
    s.foo();
} //^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn deref_trait_with_question_mark_size() {
    check_types(
        r#"
//- minicore: deref
struct Arc<T: ?Sized>;
impl<T: ?Sized> core::ops::Deref for Arc<T> {
    type Target = T;
}

struct S;
impl S {
    fn foo(&self) -> u128 { 0 }
}

fn test(s: Arc<S>) {
    (*s, s.foo());
} //^^^^^^^^^^^^^ (S, u128)
"#,
    );
}

#[test]
fn deref_trait_with_implicit_sized_requirement_on_inference_var() {
    check_types(
        r#"
//- minicore: deref
struct Foo<T>;
impl<T> core::ops::Deref for Foo<T> {
    type Target = ();
}
fn test() {
    let foo = Foo;
    *foo;
  //^^^^ ()
    let _: Foo<u8> = foo;
}
"#,
    )
}

#[test]
fn obligation_from_function_clause() {
    check_types(
        r#"
struct S;

trait Trait<T> {}
impl Trait<u32> for S {}

fn foo<T: Trait<U>, U>(t: T) -> U { loop {} }

fn test(s: S) {
    foo(s);
} //^^^^^^ u32
"#,
    );
}

#[test]
fn obligation_from_method_clause() {
    check_types(
        r#"
//- /main.rs
struct S;

trait Trait<T> {}
impl Trait<isize> for S {}

struct O;
impl O {
    fn foo<T: Trait<U>, U>(&self, t: T) -> U { loop {} }
}

fn test() {
    O.foo(S);
} //^^^^^^^^ isize
"#,
    );
}

#[test]
fn obligation_from_self_method_clause() {
    check_types(
        r#"
struct S;

trait Trait<T> {}
impl Trait<i64> for S {}

impl S {
    fn foo<U>(&self) -> U where Self: Trait<U> { loop {} }
}

fn test() {
    S.foo();
} //^^^^^^^ i64
"#,
    );
}

#[test]
fn obligation_from_impl_clause() {
    check_types(
        r#"
struct S;

trait Trait<T> {}
impl Trait<&str> for S {}

struct O<T>;
impl<U, T: Trait<U>> O<T> {
    fn foo(&self) -> U { loop {} }
}

fn test(o: O<S>) {
    o.foo();
} //^^^^^^^ &str
"#,
    );
}

#[test]
fn generic_param_env_1() {
    check_types(
        r#"
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl Clone for S {}
impl<T> Trait for T where T: Clone {}
fn test<T: Clone>(t: T) { t.foo(); }
                        //^^^^^^^ u128
"#,
    );
}

#[test]
fn generic_param_env_1_not_met() {
    check_types(
        r#"
//- /main.rs
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl Clone for S {}
impl<T> Trait for T where T: Clone {}
fn test<T>(t: T) { t.foo(); }
                 //^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn generic_param_env_2() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S {}
fn test<T: Trait>(t: T) { t.foo(); }
                        //^^^^^^^ u128
"#,
    );
}

#[test]
fn generic_param_env_2_not_met() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S {}
fn test<T>(t: T) { t.foo(); }
                 //^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn generic_param_env_deref() {
    check_types(
        r#"
//- minicore: deref
trait Trait {}
impl<T> core::ops::Deref for T where T: Trait {
    type Target = i128;
}
fn test<T: Trait>(t: T) { *t; }
                        //^^ i128
"#,
    );
}

#[test]
fn associated_type_placeholder() {
    // inside the generic function, the associated type gets normalized to a placeholder `ApplL::Out<T>` [https://rust-lang.github.io/rustc-guide/traits/associated-types.html#placeholder-associated-types].
    check_types(
        r#"
pub trait ApplyL {
    type Out;
}

pub struct RefMutL<T>;

impl<T> ApplyL for RefMutL<T> {
    type Out = <T as ApplyL>::Out;
}

fn test<T: ApplyL>() {
    let y: <RefMutL<T> as ApplyL>::Out = no_matter;
    y;
} //^ ApplyL::Out<T>
"#,
    );
}

#[test]
fn associated_type_placeholder_2() {
    check_types(
        r#"
pub trait ApplyL {
    type Out;
}
fn foo<T: ApplyL>(t: T) -> <T as ApplyL>::Out;

fn test<T: ApplyL>(t: T) {
    let y = foo(t);
    y;
} //^ ApplyL::Out<T>
"#,
    );
}

#[test]
fn argument_impl_trait() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            29..33 'self': &Self
            54..58 'self': &Self
            77..78 'x': impl Trait<u16>
            97..99 '{}': ()
            154..155 'x': impl Trait<u64>
            174..175 'y': &impl Trait<u32>
            195..323 '{     ...2(); }': ()
            201..202 'x': impl Trait<u64>
            208..209 'y': &impl Trait<u32>
            219..220 'z': S<u16>
            223..224 'S': S<u16>(u16) -> S<u16>
            223..227 'S(1)': S<u16>
            225..226 '1': u16
            233..236 'bar': fn bar(S<u16>)
            233..239 'bar(z)': ()
            237..238 'z': S<u16>
            245..246 'x': impl Trait<u64>
            245..252 'x.foo()': u64
            258..259 'y': &impl Trait<u32>
            258..265 'y.foo()': u32
            271..272 'z': S<u16>
            271..278 'z.foo()': u16
            284..285 'x': impl Trait<u64>
            284..292 'x.foo2()': i64
            298..299 'y': &impl Trait<u32>
            298..306 'y.foo2()': i64
            312..313 'z': S<u16>
            312..320 'z.foo2()': i64
        "#]],
    );
}

#[test]
fn argument_impl_trait_type_args_1() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            155..156 'x': impl Trait
            175..186 '{ loop {} }': T
            177..184 'loop {}': !
            182..184 '{}': ()
            199..200 'x': impl Trait
            219..230 '{ loop {} }': T
            221..228 'loop {}': !
            226..228 '{}': ()
            300..509 '{     ... i32 }': ()
            306..314 'Foo::bar': fn bar<{unknown}, {unknown}>(S) -> {unknown}
            306..317 'Foo::bar(S)': {unknown}
            315..316 'S': S
            323..338 '<F as Foo>::bar': fn bar<F, {unknown}>(S) -> {unknown}
            323..341 '<F as ...bar(S)': {unknown}
            339..340 'S': S
            347..353 'F::bar': fn bar<F, {unknown}>(S) -> {unknown}
            347..356 'F::bar(S)': {unknown}
            354..355 'S': S
            362..377 'Foo::bar::<u32>': fn bar<{unknown}, u32>(S) -> u32
            362..380 'Foo::b...32>(S)': u32
            378..379 'S': S
            386..408 '<F as ...:<u32>': fn bar<F, u32>(S) -> u32
            386..411 '<F as ...32>(S)': u32
            409..410 'S': S
            418..421 'foo': fn foo<{unknown}>(S) -> {unknown}
            418..424 'foo(S)': {unknown}
            422..423 'S': S
            430..440 'foo::<u32>': fn foo<u32>(S) -> u32
            430..443 'foo::<u32>(S)': u32
            441..442 'S': S
            449..464 'foo::<u32, i32>': fn foo<u32>(S) -> u32
            449..467 'foo::<...32>(S)': u32
            465..466 'S': S
        "#]],
    );
}

#[test]
fn argument_impl_trait_type_args_2() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            87..91 'self': F<T>
            93..94 'x': impl Trait
            118..129 '{ loop {} }': (T, U)
            120..127 'loop {}': !
            125..127 '{}': ()
            143..283 '{     ...ored }': ()
            149..150 'F': F<{unknown}>
            149..157 'F.foo(S)': ({unknown}, {unknown})
            155..156 'S': S
            163..171 'F::<u32>': F<u32>
            163..178 'F::<u32>.foo(S)': (u32, {unknown})
            176..177 'S': S
            184..192 'F::<u32>': F<u32>
            184..206 'F::<u3...32>(S)': (u32, i32)
            204..205 'S': S
            212..220 'F::<u32>': F<u32>
            212..239 'F::<u3...32>(S)': (u32, i32)
            237..238 'S': S
        "#]],
    );
}

#[test]
fn argument_impl_trait_to_fn_pointer() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
trait Trait {}
fn foo(x: impl Trait) { loop {} }
struct S;
impl Trait for S {}

fn test() {
    let f: fn(S) -> () = foo;
}"#,
        expect![[r#"
            22..23 'x': impl Trait
            37..48 '{ loop {} }': ()
            39..46 'loop {}': !
            44..46 '{}': ()
            90..123 '{     ...foo; }': ()
            100..101 'f': fn(S)
            117..120 'foo': fn foo(S)
        "#]],
    );
}

#[test]
fn impl_trait() {
    check_infer(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            29..33 'self': &Self
            54..58 'self': &Self
            98..100 '{}': ()
            110..111 'x': impl Trait<u64>
            130..131 'y': &impl Trait<u64>
            151..268 '{     ...2(); }': ()
            157..158 'x': impl Trait<u64>
            164..165 'y': &impl Trait<u64>
            175..176 'z': impl Trait<u64>
            179..182 'bar': fn bar() -> impl Trait<u64>
            179..184 'bar()': impl Trait<u64>
            190..191 'x': impl Trait<u64>
            190..197 'x.foo()': u64
            203..204 'y': &impl Trait<u64>
            203..210 'y.foo()': u64
            216..217 'z': impl Trait<u64>
            216..223 'z.foo()': u64
            229..230 'x': impl Trait<u64>
            229..237 'x.foo2()': i64
            243..244 'y': &impl Trait<u64>
            243..251 'y.foo2()': i64
            257..258 'z': impl Trait<u64>
            257..265 'z.foo2()': i64
        "#]],
    );
}

#[test]
fn simple_return_pos_impl_trait() {
    cov_mark::check!(lower_rpit);
    check_infer(
        r#"
//- minicore: sized
trait Trait<T> {
    fn foo(&self) -> T;
}
fn bar() -> impl Trait<u64> { loop {} }

fn test() {
    let a = bar();
    a.foo();
}"#,
        expect![[r#"
            29..33 'self': &Self
            71..82 '{ loop {} }': !
            73..80 'loop {}': !
            78..80 '{}': ()
            94..129 '{     ...o(); }': ()
            104..105 'a': impl Trait<u64>
            108..111 'bar': fn bar() -> impl Trait<u64>
            108..113 'bar()': impl Trait<u64>
            119..120 'a': impl Trait<u64>
            119..126 'a.foo()': u64
        "#]],
    );
}

#[test]
fn more_return_pos_impl_trait() {
    check_infer(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            49..53 'self': &mut Self
            101..105 'self': &Self
            184..195 '{ loop {} }': ({unknown}, {unknown})
            186..193 'loop {}': !
            191..193 '{}': ()
            206..207 't': T
            268..279 '{ loop {} }': ({unknown}, {unknown})
            270..277 'loop {}': !
            275..277 '{}': ()
            291..413 '{     ...o(); }': ()
            301..307 '(a, b)': (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>)
            302..303 'a': impl Iterator<Item = impl Trait<u32>>
            305..306 'b': impl Trait<u64>
            310..313 'bar': fn bar() -> (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>)
            310..315 'bar()': (impl Iterator<Item = impl Trait<u32>>, impl Trait<u64>)
            321..322 'a': impl Iterator<Item = impl Trait<u32>>
            321..329 'a.next()': impl Trait<u32>
            321..335 'a.next().foo()': u32
            341..342 'b': impl Trait<u64>
            341..348 'b.foo()': u64
            358..364 '(c, d)': (impl Iterator<Item = impl Trait<u128>>, impl Trait<u128>)
            359..360 'c': impl Iterator<Item = impl Trait<u128>>
            362..363 'd': impl Trait<u128>
            367..370 'baz': fn baz<u128>(u128) -> (impl Iterator<Item = impl Trait<u128>>, impl Trait<u128>)
            367..377 'baz(1u128)': (impl Iterator<Item = impl Trait<u128>>, impl Trait<u128>)
            371..376 '1u128': u128
            383..384 'c': impl Iterator<Item = impl Trait<u128>>
            383..391 'c.next()': impl Trait<u128>
            383..397 'c.next().foo()': u128
            403..404 'd': impl Trait<u128>
            403..410 'd.foo()': u128
        "#]],
    );
}

#[test]
fn infer_from_return_pos_impl_trait() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn, sized
trait Trait<T> {}
struct Bar<T>(T);
impl<T> Trait<T> for Bar<T> {}
fn foo<const C: u8, T>() -> (impl FnOnce(&str, T), impl Trait<u8>) {
    (|input, t| {}, Bar(C))
}
"#,
        expect![[r#"
            134..165 '{     ...(C)) }': (|&str, T| -> (), Bar<u8>)
            140..163 '(|inpu...ar(C))': (|&str, T| -> (), Bar<u8>)
            141..154 '|input, t| {}': |&str, T| -> ()
            142..147 'input': &str
            149..150 't': T
            152..154 '{}': ()
            156..159 'Bar': Bar<u8>(u8) -> Bar<u8>
            156..162 'Bar(C)': Bar<u8>
            160..161 'C': u8
        "#]],
    );
}

#[test]
fn return_pos_impl_trait_in_projection() {
    // Note that the unused type param `X` is significant; see #13307.
    check_no_mismatches(
        r#"
//- minicore: sized
trait Future { type Output; }
impl Future for () { type Output = i32; }
type Foo<F> = (<F as Future>::Output, F);
fn foo<X>() -> Foo<impl Future<Output = ()>> {
    (0, ())
}
"#,
    )
}

#[test]
fn dyn_trait() {
    check_infer(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            29..33 'self': &Self
            54..58 'self': &Self
            97..99 '{}': dyn Trait<u64>
            109..110 'x': dyn Trait<u64>
            128..129 'y': &dyn Trait<u64>
            148..265 '{     ...2(); }': ()
            154..155 'x': dyn Trait<u64>
            161..162 'y': &dyn Trait<u64>
            172..173 'z': dyn Trait<u64>
            176..179 'bar': fn bar() -> dyn Trait<u64>
            176..181 'bar()': dyn Trait<u64>
            187..188 'x': dyn Trait<u64>
            187..194 'x.foo()': u64
            200..201 'y': &dyn Trait<u64>
            200..207 'y.foo()': u64
            213..214 'z': dyn Trait<u64>
            213..220 'z.foo()': u64
            226..227 'x': dyn Trait<u64>
            226..234 'x.foo2()': i64
            240..241 'y': &dyn Trait<u64>
            240..248 'y.foo2()': i64
            254..255 'z': dyn Trait<u64>
            254..262 'z.foo2()': i64
        "#]],
    );
}

#[test]
fn dyn_trait_in_impl() {
    check_infer(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            32..36 'self': &Self
            102..106 'self': &S<T, U>
            128..139 '{ loop {} }': &dyn Trait<T, U>
            130..137 'loop {}': !
            135..137 '{}': ()
            175..179 'self': &Self
            251..252 's': S<u32, i32>
            267..289 '{     ...z(); }': ()
            273..274 's': S<u32, i32>
            273..280 's.bar()': &dyn Trait<u32, i32>
            273..286 's.bar().baz()': (u32, i32)
        "#]],
    );
}

#[test]
fn dyn_trait_bare() {
    check_infer(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            26..30 'self': &Self
            60..62 '{}': dyn Trait
            72..73 'x': dyn Trait
            82..83 'y': &dyn Trait
            100..175 '{     ...o(); }': u64
            106..107 'x': dyn Trait
            113..114 'y': &dyn Trait
            124..125 'z': dyn Trait
            128..131 'bar': fn bar() -> dyn Trait
            128..133 'bar()': dyn Trait
            139..140 'x': dyn Trait
            139..146 'x.foo()': u64
            152..153 'y': &dyn Trait
            152..159 'y.foo()': u64
            165..166 'z': dyn Trait
            165..172 'z.foo()': u64
        "#]],
    );

    check_infer_with_mismatches(
        r#"
//- minicore: fn, coerce_unsized
struct S;
impl S {
    fn foo(&self) {}
}
fn f(_: &Fn(S)) {}
fn main() {
    f(&|number| number.foo());
}
        "#,
        expect![[r#"
            31..35 'self': &S
            37..39 '{}': ()
            47..48 '_': &dyn Fn(S)
            58..60 '{}': ()
            71..105 '{     ...()); }': ()
            77..78 'f': fn f(&dyn Fn(S))
            77..102 'f(&|nu...foo())': ()
            79..101 '&|numb....foo()': &|S| -> ()
            80..101 '|numbe....foo()': |S| -> ()
            81..87 'number': S
            89..95 'number': S
            89..101 'number.foo()': ()
        "#]],
    )
}

#[test]
fn weird_bounds() {
    check_infer(
        r#"
//- minicore: sized
trait Trait {}
fn test(
    a: impl Trait + 'lifetime,
    b: impl 'lifetime,
    c: impl (Trait),
    d: impl ('lifetime),
    e: impl ?Sized,
    f: impl Trait + ?Sized
) {}
"#,
        expect![[r#"
            28..29 'a': impl Trait
            59..60 'b': impl Sized
            82..83 'c': impl Trait
            103..104 'd': impl Sized
            128..129 'e': impl ?Sized
            148..149 'f': impl Trait + ?Sized
            173..175 '{}': ()
        "#]],
    );
}

#[test]
fn error_bound_chalk() {
    check_types(
        r#"
trait Trait {
    fn foo(&self) -> u32 { 0 }
}

fn test(x: (impl Trait + UnknownTrait)) {
    x.foo();
} //^^^^^^^ u32
"#,
    );
}

#[test]
fn assoc_type_bindings() {
    check_infer(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            49..50 't': T
            77..79 '{}': Trait::Type<T>
            111..112 't': T
            122..124 '{}': U
            154..155 't': T
            165..168 '{t}': T
            166..167 't': T
            256..257 'x': T
            262..263 'y': impl Trait<Type = i64>
            289..397 '{     ...r>); }': ()
            295..298 'get': fn get<T>(T) -> <T as Trait>::Type
            295..301 'get(x)': u32
            299..300 'x': T
            307..311 'get2': fn get2<u32, T>(T) -> u32
            307..314 'get2(x)': u32
            312..313 'x': T
            320..323 'get': fn get<impl Trait<Type = i64>>(impl Trait<Type = i64>) -> <impl Trait<Type = i64> as Trait>::Type
            320..326 'get(y)': i64
            324..325 'y': impl Trait<Type = i64>
            332..336 'get2': fn get2<i64, impl Trait<Type = i64>>(impl Trait<Type = i64>) -> i64
            332..339 'get2(y)': i64
            337..338 'y': impl Trait<Type = i64>
            345..348 'get': fn get<S<u64>>(S<u64>) -> <S<u64> as Trait>::Type
            345..356 'get(set(S))': u64
            349..352 'set': fn set<S<u64>>(S<u64>) -> S<u64>
            349..355 'set(S)': S<u64>
            353..354 'S': S<u64>
            362..366 'get2': fn get2<u64, S<u64>>(S<u64>) -> u64
            362..374 'get2(set(S))': u64
            367..370 'set': fn set<S<u64>>(S<u64>) -> S<u64>
            367..373 'set(S)': S<u64>
            371..372 'S': S<u64>
            380..384 'get2': fn get2<str, S<str>>(S<str>) -> str
            380..394 'get2(S::<str>)': str
            385..393 'S::<str>': S<str>
        "#]],
    );
}

#[test]
fn impl_trait_assoc_binding_projection_bug() {
    check_types(
        r#"
//- minicore: iterator
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
        node.clone();
    } //^^^^^^^^^^^^ {unknown}
}
"#,
    );
}

#[test]
fn projection_eq_within_chalk() {
    check_infer(
        r#"
trait Trait1 {
    type Type;
}
trait Trait2<T> {
    fn foo(self) -> T;
}
impl<T, U> Trait2<T> for U where U: Trait1<Type = T> {}

fn test<T: Trait1<Type = u32>>(x: T) {
    x.foo();
}"#,
        expect![[r#"
            61..65 'self': Self
            163..164 'x': T
            169..185 '{     ...o(); }': ()
            175..176 'x': T
            175..182 'x.foo()': u32
        "#]],
    );
}

#[test]
fn where_clause_trait_in_scope_for_method_resolution() {
    check_types(
        r#"
mod foo {
    pub trait Trait {
        fn foo(&self) -> u32 { 0 }
    }
}

fn test<T: foo::Trait>(x: T) {
    x.foo();
} //^^^^^^^ u32
"#,
    );
}

#[test]
fn super_trait_method_resolution() {
    check_infer(
        r#"
mod foo {
    pub trait SuperTrait {
        fn foo(&self) -> u32 {}
    }
}
trait Trait1: foo::SuperTrait {}
trait Trait2 where Self: foo::SuperTrait {}

fn test<T: Trait1, U: Trait2>(x: T, y: U) {
    x.foo();
    y.foo();
}"#,
        expect![[r#"
            53..57 'self': &Self
            66..68 '{}': u32
            185..186 'x': T
            191..192 'y': U
            197..226 '{     ...o(); }': ()
            203..204 'x': T
            203..210 'x.foo()': u32
            216..217 'y': U
            216..223 'y.foo()': u32
        "#]],
    );
}

#[test]
fn super_trait_impl_trait_method_resolution() {
    check_infer(
        r#"
//- minicore: sized
mod foo {
    pub trait SuperTrait {
        fn foo(&self) -> u32 {}
    }
}
trait Trait1: foo::SuperTrait {}

fn test(x: &impl Trait1) {
    x.foo();
}"#,
        expect![[r#"
            53..57 'self': &Self
            66..68 '{}': u32
            119..120 'x': &impl Trait1
            136..152 '{     ...o(); }': ()
            142..143 'x': &impl Trait1
            142..149 'x.foo()': u32
        "#]],
    );
}

#[test]
fn super_trait_cycle() {
    // This just needs to not crash
    check_infer(
        r#"
        trait A: B {}
        trait B: A {}

        fn test<T: A>(x: T) {
            x.foo();
        }
        "#,
        expect![[r#"
            43..44 'x': T
            49..65 '{     ...o(); }': ()
            55..56 'x': T
            55..62 'x.foo()': {unknown}
        "#]],
    );
}

#[test]
fn super_trait_assoc_type_bounds() {
    check_infer(
        r#"
trait SuperTrait { type Type; }
trait Trait where Self: SuperTrait {}

fn get2<U, T: Trait<Type = U>>(t: T) -> U {}
fn set<T: Trait<Type = u64>>(t: T) -> T {t}

struct S<T>;
impl<T> SuperTrait for S<T> { type Type = T; }
impl<T> Trait for S<T> {}

fn test() {
    get2(set(S));
}"#,
        expect![[r#"
            102..103 't': T
            113..115 '{}': U
            145..146 't': T
            156..159 '{t}': T
            157..158 't': T
            258..279 '{     ...S)); }': ()
            264..268 'get2': fn get2<u64, S<u64>>(S<u64>) -> u64
            264..276 'get2(set(S))': u64
            269..272 'set': fn set<S<u64>>(S<u64>) -> S<u64>
            269..275 'set(S)': S<u64>
            273..274 'S': S<u64>
        "#]],
    );
}

#[test]
fn fn_trait() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn

fn test<F: FnOnce(u32, u64) -> u128>(f: F) {
    f.call_once((1, 2));
}"#,
        expect![[r#"
            38..39 'f': F
            44..72 '{     ...2)); }': ()
            50..51 'f': F
            50..69 'f.call...1, 2))': u128
            62..68 '(1, 2)': (u32, u64)
            63..64 '1': u32
            66..67 '2': u64
        "#]],
    );
}

#[test]
fn fn_ptr_and_item() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn

trait Foo<T> {
    fn foo(&self) -> T;
}

struct Bar<T>(T);

impl<A1, R, F: FnOnce(A1) -> R> Foo<(A1, R)> for Bar<F> {
    fn foo(&self) -> (A1, R) { loop {} }
}

enum Opt<T> { None, Some(T) }
impl<T> Opt<T> {
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Opt<U> { loop {} }
}

fn test() {
    let bar: Bar<fn(u8) -> u32>;
    bar.foo();

    let opt: Opt<u8>;
    let f: fn(u8) -> u32;
    opt.map(f);
}"#,
        expect![[r#"
            28..32 'self': &Self
            132..136 'self': &Bar<F>
            149..160 '{ loop {} }': (A1, R)
            151..158 'loop {}': !
            156..158 '{}': ()
            244..248 'self': Opt<T>
            250..251 'f': F
            266..277 '{ loop {} }': Opt<U>
            268..275 'loop {}': !
            273..275 '{}': ()
            291..407 '{     ...(f); }': ()
            301..304 'bar': Bar<fn(u8) -> u32>
            330..333 'bar': Bar<fn(u8) -> u32>
            330..339 'bar.foo()': (u8, u32)
            350..353 'opt': Opt<u8>
            372..373 'f': fn(u8) -> u32
            394..397 'opt': Opt<u8>
            394..404 'opt.map(f)': Opt<u32>
            402..403 'f': fn(u8) -> u32
        "#]],
    );
}

#[test]
fn fn_trait_deref_with_ty_default() {
    check_infer(
        r#"
//- minicore: deref, fn
struct Foo;

impl Foo {
    fn foo(&self) -> usize {}
}

struct Lazy<T, F = fn() -> T>(F);

impl<T, F> Lazy<T, F> {
    pub fn new(f: F) -> Lazy<T, F> {}
}

impl<T, F: FnOnce() -> T> core::ops::Deref for Lazy<T, F> {
    type Target = T;
}

fn test() {
    let lazy1: Lazy<Foo, _> = Lazy::new(|| Foo);
    let r1 = lazy1.foo();

    fn make_foo_fn() -> Foo {}
    let make_foo_fn_ptr: fn() -> Foo = make_foo_fn;
    let lazy2: Lazy<Foo, _> = Lazy::new(make_foo_fn_ptr);
    let r2 = lazy2.foo();
}"#,
        expect![[r#"
            36..40 'self': &Foo
            51..53 '{}': usize
            131..132 'f': F
            151..153 '{}': Lazy<T, F>
            251..497 '{     ...o(); }': ()
            261..266 'lazy1': Lazy<Foo, || -> Foo>
            283..292 'Lazy::new': fn new<Foo, || -> Foo>(|| -> Foo) -> Lazy<Foo, || -> Foo>
            283..300 'Lazy::...| Foo)': Lazy<Foo, || -> Foo>
            293..299 '|| Foo': || -> Foo
            296..299 'Foo': Foo
            310..312 'r1': usize
            315..320 'lazy1': Lazy<Foo, || -> Foo>
            315..326 'lazy1.foo()': usize
            368..383 'make_foo_fn_ptr': fn() -> Foo
            399..410 'make_foo_fn': fn make_foo_fn() -> Foo
            420..425 'lazy2': Lazy<Foo, fn() -> Foo>
            442..451 'Lazy::new': fn new<Foo, fn() -> Foo>(fn() -> Foo) -> Lazy<Foo, fn() -> Foo>
            442..468 'Lazy::...n_ptr)': Lazy<Foo, fn() -> Foo>
            452..467 'make_foo_fn_ptr': fn() -> Foo
            478..480 'r2': usize
            483..488 'lazy2': Lazy<Foo, fn() -> Foo>
            483..494 'lazy2.foo()': usize
            357..359 '{}': Foo
        "#]],
    );
}

#[test]
fn closure_1() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn
enum Option<T> { Some(T), None }
impl<T> Option<T> {
    fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U> { loop {} }
}

fn test() {
    let x = Option::Some(1u32);
    x.map(|v| v + 1);
    x.map(|_v| 1u64);
    let y: Option<i64> = x.map(|_v| 1);
}"#,
        expect![[r#"
            86..90 'self': Option<T>
            92..93 'f': F
            111..122 '{ loop {} }': Option<U>
            113..120 'loop {}': !
            118..120 '{}': ()
            136..255 '{     ... 1); }': ()
            146..147 'x': Option<u32>
            150..162 'Option::Some': Some<u32>(u32) -> Option<u32>
            150..168 'Option...(1u32)': Option<u32>
            163..167 '1u32': u32
            174..175 'x': Option<u32>
            174..190 'x.map(...v + 1)': Option<u32>
            180..189 '|v| v + 1': |u32| -> u32
            181..182 'v': u32
            184..185 'v': u32
            184..189 'v + 1': u32
            188..189 '1': u32
            196..197 'x': Option<u32>
            196..212 'x.map(... 1u64)': Option<u64>
            202..211 '|_v| 1u64': |u32| -> u64
            203..205 '_v': u32
            207..211 '1u64': u64
            222..223 'y': Option<i64>
            239..240 'x': Option<u32>
            239..252 'x.map(|_v| 1)': Option<i64>
            245..251 '|_v| 1': |u32| -> i64
            246..248 '_v': u32
            250..251 '1': i64
        "#]],
    );
}

#[test]
fn closure_2() {
    check_types(
        r#"
//- minicore: add, fn

impl core::ops::Add for u64 {
    type Output = Self;
    fn add(self, rhs: u64) -> Self::Output {0}
}

impl core::ops::Add for u128 {
    type Output = Self;
    fn add(self, rhs: u128) -> Self::Output {0}
}

fn test<F: FnOnce(u32) -> u64>(f: F) {
    f(1);
  //  ^ u32
  //^^^^ u64
    let g = |v| v + 1;
              //^^^^^ u64
          //^^^^^^^^^ |u64| -> u64
    g(1u64);
  //^^^^^^^ u64
    let h = |v| 1u128 + v;
          //^^^^^^^^^^^^^ |u128| -> u128
}"#,
    );
}

#[test]
fn closure_as_argument_inference_order() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn
fn foo1<T, U, F: FnOnce(T) -> U>(x: T, f: F) -> U { loop {} }
fn foo2<T, U, F: FnOnce(T) -> U>(f: F, x: T) -> U { loop {} }

struct S;
impl S {
    fn method(self) -> u64;

    fn foo1<T, U, F: FnOnce(T) -> U>(self, x: T, f: F) -> U { loop {} }
    fn foo2<T, U, F: FnOnce(T) -> U>(self, f: F, x: T) -> U { loop {} }
}

fn test() {
    let x1 = foo1(S, |s| s.method());
    let x2 = foo2(|s| s.method(), S);
    let x3 = S.foo1(S, |s| s.method());
    let x4 = S.foo2(|s| s.method(), S);
}"#,
        expect![[r#"
            33..34 'x': T
            39..40 'f': F
            50..61 '{ loop {} }': U
            52..59 'loop {}': !
            57..59 '{}': ()
            95..96 'f': F
            101..102 'x': T
            112..123 '{ loop {} }': U
            114..121 'loop {}': !
            119..121 '{}': ()
            158..162 'self': S
            210..214 'self': S
            216..217 'x': T
            222..223 'f': F
            233..244 '{ loop {} }': U
            235..242 'loop {}': !
            240..242 '{}': ()
            282..286 'self': S
            288..289 'f': F
            294..295 'x': T
            305..316 '{ loop {} }': U
            307..314 'loop {}': !
            312..314 '{}': ()
            330..489 '{     ... S); }': ()
            340..342 'x1': u64
            345..349 'foo1': fn foo1<S, u64, |S| -> u64>(S, |S| -> u64) -> u64
            345..368 'foo1(S...hod())': u64
            350..351 'S': S
            353..367 '|s| s.method()': |S| -> u64
            354..355 's': S
            357..358 's': S
            357..367 's.method()': u64
            378..380 'x2': u64
            383..387 'foo2': fn foo2<S, u64, |S| -> u64>(|S| -> u64, S) -> u64
            383..406 'foo2(|...(), S)': u64
            388..402 '|s| s.method()': |S| -> u64
            389..390 's': S
            392..393 's': S
            392..402 's.method()': u64
            404..405 'S': S
            416..418 'x3': u64
            421..422 'S': S
            421..446 'S.foo1...hod())': u64
            428..429 'S': S
            431..445 '|s| s.method()': |S| -> u64
            432..433 's': S
            435..436 's': S
            435..445 's.method()': u64
            456..458 'x4': u64
            461..462 'S': S
            461..486 'S.foo2...(), S)': u64
            468..482 '|s| s.method()': |S| -> u64
            469..470 's': S
            472..473 's': S
            472..482 's.method()': u64
            484..485 'S': S
        "#]],
    );
}

#[test]
fn fn_item_fn_trait() {
    check_types(
        r#"
//- minicore: fn
struct S;

fn foo() -> S { S }

fn takes_closure<U, F: FnOnce() -> U>(f: F) -> U { f() }

fn test() {
    takes_closure(foo);
} //^^^^^^^^^^^^^^^^^^ S
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_1() {
    check_types(
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
    x.foo();
} //^^^^^^^ u32
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_2() {
    check_types(
        r#"
trait Trait<T> {
    type Item;
}

trait Trait2 {
    fn foo(&self) -> u32;
}

fn test<T, U>() where T::Item: Trait2, T: Trait<U::Item>, U: Trait<()> {
    let x: T::Item = no_matter;
    x.foo();
} //^^^^^^^ u32
"#,
    );
}

#[test]
fn unselected_projection_on_impl_self() {
    check_infer(
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
}"#,
        expect![[r#"
            40..44 'self': &Self
            46..47 'x': Trait::Item<Self>
            126..130 'self': &S
            132..133 'x': u32
            147..161 '{ let y = x; }': ()
            153..154 'y': u32
            157..158 'x': u32
            228..232 'self': &S2
            234..235 'x': i32
            251..265 '{ let y = x; }': ()
            257..258 'y': i32
            261..262 'x': i32
        "#]],
    );
}

#[test]
fn unselected_projection_on_trait_self() {
    check_types(
        r#"
trait Trait {
    type Item;

    fn f(&self) -> Self::Item { loop {} }
}

struct S;
impl Trait for S {
    type Item = u32;
}

fn test() {
    S.f();
} //^^^^^ u32
"#,
    );
}

#[test]
fn unselected_projection_chalk_fold() {
    check_types(
        r#"
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
    fold(interner, t);
} //^^^^^^^^^^^^^^^^^ Ty<I>
"#,
    );
}

#[test]
fn trait_impl_self_ty() {
    check_types(
        r#"
trait Trait<T> {
   fn foo(&self);
}

struct S;

impl Trait<Self> for S {}

fn test() {
    S.foo();
} //^^^^^^^ ()
"#,
    );
}

#[test]
fn trait_impl_self_ty_cycle() {
    check_types(
        r#"
trait Trait {
   fn foo(&self);
}

struct S<T>;

impl Trait for S<Self> {}

fn test() {
    S.foo();
} //^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_cycle_1() {
    // This is not a cycle, because the `T: Trait2<T::Item>` bound depends only on the `T: Trait`
    // bound, not on itself (since only `Trait` can define `Item`).
    check_types(
        r#"
trait Trait {
    type Item;
}

trait Trait2<T> {}

fn test<T: Trait>() where T: Trait2<T::Item> {
    let x: T::Item = no_matter;
}                  //^^^^^^^^^ Trait::Item<T>
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_cycle_2() {
    // this is a legitimate cycle
    check_types(
        r#"
//- /main.rs
trait Trait<T> {
    type Item;
}

fn test<T, U>() where T: Trait<U::Item>, U: Trait<T::Item> {
    let x: T::Item = no_matter;
}                  //^^^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_cycle_3() {
    // this is a cycle for rustc; we currently accept it
    check_types(
        r#"
//- /main.rs
trait Trait {
    type Item;
    type OtherItem;
}

fn test<T>() where T: Trait<OtherItem = T::Item> {
    let x: T::Item = no_matter;
}                  //^^^^^^^^^ Trait::Item<T>
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_no_cycle() {
    // this is not a cycle
    check_types(
        r#"
//- minicore: index
use core::ops::Index;

type Key<S: UnificationStoreBase> = <S as UnificationStoreBase>::Key;

pub trait UnificationStoreBase: Index<Output = Key<Self>> {
    type Key;

    fn len(&self) -> usize;
}

pub trait UnificationStoreMut: UnificationStoreBase {
    fn push(&mut self, value: Self::Key);
}

fn test<T>(t: T) where T: UnificationStoreMut {
    let x;
    t.push(x);
    let y: Key<T>;
    (x, y);
} //^^^^^^ (UnificationStoreBase::Key<T>, UnificationStoreBase::Key<T>)
"#,
    );
}

#[test]
fn inline_assoc_type_bounds_1() {
    check_types(
        r#"
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
    x.foo();
} //^^^^^^^ u32
"#,
    );
}

#[test]
fn inline_assoc_type_bounds_2() {
    check_types(
        r#"
trait Iterator {
    type Item;
}

fn test<I: Iterator<Item: Iterator<Item = u32>>>() {
    let x: <<I as Iterator>::Item as Iterator>::Item;
    x;
} //^ u32
"#,
    );
}

#[test]
fn proc_macro_server_types() {
    check_infer(
        r#"
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
struct RustAnalyzer;
impl Types for RustAnalyzer {
    type TokenStream = T;
    type Group = G;
}

fn make<T>() -> T { loop {} }
impl TokenStream for RustAnalyzer {
    fn new() -> Self::TokenStream {
        let group: Self::Group = make();
        make()
    }
}"#,
        expect![[r#"
            1075..1086 '{ loop {} }': T
            1077..1084 'loop {}': !
            1082..1084 '{}': ()
            1157..1220 '{     ...     }': T
            1171..1176 'group': G
            1192..1196 'make': fn make<G>() -> G
            1192..1198 'make()': G
            1208..1212 'make': fn make<T>() -> T
            1208..1214 'make()': T
        "#]],
    );
}

#[test]
fn unify_impl_trait() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
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
}"#,
        expect![[r#"
            26..27 'x': impl Trait<u32>
            46..57 '{ loop {} }': ()
            48..55 'loop {}': !
            53..55 '{}': ()
            68..69 'x': impl Trait<T>
            91..102 '{ loop {} }': T
            93..100 'loop {}': !
            98..100 '{}': ()
            171..182 '{ loop {} }': T
            173..180 'loop {}': !
            178..180 '{}': ()
            213..309 '{     ...t()) }': S<i32>
            223..225 's1': S<u32>
            228..229 'S': S<u32>(u32) -> S<u32>
            228..240 'S(default())': S<u32>
            230..237 'default': fn default<u32>() -> u32
            230..239 'default()': u32
            246..249 'foo': fn foo(S<u32>)
            246..253 'foo(s1)': ()
            250..252 's1': S<u32>
            263..264 'x': i32
            272..275 'bar': fn bar<i32>(S<i32>) -> i32
            272..289 'bar(S(...lt()))': i32
            276..277 'S': S<i32>(i32) -> S<i32>
            276..288 'S(default())': S<i32>
            278..285 'default': fn default<i32>() -> i32
            278..287 'default()': i32
            295..296 'S': S<i32>(i32) -> S<i32>
            295..307 'S(default())': S<i32>
            297..304 'default': fn default<i32>() -> i32
            297..306 'default()': i32
        "#]],
    );
}

#[test]
fn assoc_types_from_bounds() {
    check_infer(
        r#"
//- minicore: fn
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
}"#,
        expect![[r#"
            72..74 '_v': F
            117..120 '{ }': ()
            132..163 '{     ... }); }': ()
            138..148 'f::<(), _>': fn f<(), |&()| -> ()>(|&()| -> ())
            138..160 'f::<()... z; })': ()
            149..159 '|z| { z; }': |&()| -> ()
            150..151 'z': &()
            153..159 '{ z; }': ()
            155..156 'z': &()
        "#]],
    );
}

#[test]
fn associated_type_bound() {
    check_types(
        r#"
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
    y.foo();
} //^^^^^^^ u32
"#,
    );
}

#[test]
fn dyn_trait_through_chalk() {
    check_types(
        r#"
//- minicore: deref
struct Box<T: ?Sized> {}
impl<T: ?Sized> core::ops::Deref for Box<T> {
    type Target = T;
}
trait Trait {
    fn foo(&self);
}

fn test(x: Box<dyn Trait>) {
    x.foo();
} //^^^^^^^ ()
"#,
    );
}

#[test]
fn string_to_owned() {
    check_types(
        r#"
struct String {}
pub trait ToOwned {
    type Owned;
    fn to_owned(&self) -> Self::Owned;
}
impl ToOwned for str {
    type Owned = String;
}
fn test() {
    "foo".to_owned();
} //^^^^^^^^^^^^^^^^ String
"#,
    );
}

#[test]
fn iterator_chain() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn, option
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
}"#,
        expect![[r#"
            61..65 'self': Self
            67..68 'f': F
            152..163 '{ loop {} }': FilterMap<Self, F>
            154..161 'loop {}': !
            159..161 '{}': ()
            184..188 'self': Self
            190..191 'f': F
            240..251 '{ loop {} }': ()
            242..249 'loop {}': !
            247..249 '{}': ()
            360..364 'self': Self
            689..693 'self': I
            700..720 '{     ...     }': I
            710..714 'self': I
            779..790 '{ loop {} }': Vec<T>
            781..788 'loop {}': !
            786..788 '{}': ()
            977..1104 '{     ... }); }': ()
            983..998 'Vec::<i32>::new': fn new<i32>() -> Vec<i32>
            983..1000 'Vec::<...:new()': Vec<i32>
            983..1012 'Vec::<...iter()': IntoIter<i32>
            983..1075 'Vec::<...one })': FilterMap<IntoIter<i32>, |i32| -> Option<u32>>
            983..1101 'Vec::<... y; })': ()
            1029..1074 '|x| if...None }': |i32| -> Option<u32>
            1030..1031 'x': i32
            1033..1074 'if x >...None }': Option<u32>
            1036..1037 'x': i32
            1036..1041 'x > 0': bool
            1040..1041 '0': i32
            1042..1060 '{ Some...u32) }': Option<u32>
            1044..1048 'Some': Some<u32>(u32) -> Option<u32>
            1044..1058 'Some(x as u32)': Option<u32>
            1049..1050 'x': i32
            1049..1057 'x as u32': u32
            1066..1074 '{ None }': Option<u32>
            1068..1072 'None': Option<u32>
            1090..1100 '|y| { y; }': |u32| -> ()
            1091..1092 'y': u32
            1094..1100 '{ y; }': ()
            1096..1097 'y': u32
        "#]],
    );
}

#[test]
fn nested_assoc() {
    check_types(
        r#"
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
    Bar::foo();
} //^^^^^^^^^^ Foo
"#,
    );
}

#[test]
fn trait_object_no_coercion() {
    check_infer_with_mismatches(
        r#"
trait Foo {}

fn foo(x: &dyn Foo) {}

fn test(x: &dyn Foo) {
    foo(x);
}"#,
        expect![[r#"
            21..22 'x': &dyn Foo
            34..36 '{}': ()
            46..47 'x': &dyn Foo
            59..74 '{     foo(x); }': ()
            65..68 'foo': fn foo(&dyn Foo)
            65..71 'foo(x)': ()
            69..70 'x': &dyn Foo
        "#]],
    );
}

#[test]
fn builtin_copy() {
    check_infer_with_mismatches(
        r#"
//- minicore: copy
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
}"#,
        expect![[r#"
            78..82 'self': &Self
            134..235 '{     ...t(); }': ()
            140..146 'IsCopy': IsCopy
            140..153 'IsCopy.test()': bool
            159..166 'NotCopy': NotCopy
            159..173 'NotCopy.test()': {unknown}
            179..195 '(IsCop...sCopy)': (IsCopy, IsCopy)
            179..202 '(IsCop...test()': bool
            180..186 'IsCopy': IsCopy
            188..194 'IsCopy': IsCopy
            208..225 '(IsCop...tCopy)': (IsCopy, NotCopy)
            208..232 '(IsCop...test()': {unknown}
            209..215 'IsCopy': IsCopy
            217..224 'NotCopy': NotCopy
        "#]],
    );
}

#[test]
fn builtin_fn_def_copy() {
    check_infer_with_mismatches(
        r#"
//- minicore: copy
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
}"#,
        expect![[r#"
            9..11 '{}': ()
            28..29 'T': {unknown}
            36..38 '{}': T
            36..38: expected T, got ()
            113..117 'self': &Self
            169..249 '{     ...t(); }': ()
            175..178 'foo': fn foo()
            175..185 'foo.test()': bool
            191..194 'bar': fn bar<{unknown}>({unknown}) -> {unknown}
            191..201 'bar.test()': bool
            207..213 'Struct': Struct(usize) -> Struct
            207..220 'Struct.test()': bool
            226..239 'Enum::Variant': Variant(usize) -> Enum
            226..246 'Enum::...test()': bool
        "#]],
    );
}

#[test]
fn builtin_fn_ptr_copy() {
    check_infer_with_mismatches(
        r#"
//- minicore: copy
trait Test { fn test(&self) -> bool; }
impl<T: Copy> Test for T {}

fn test(f1: fn(), f2: fn(usize) -> u8, f3: fn(u8, u8) -> &u8) {
    f1.test();
    f2.test();
    f3.test();
}"#,
        expect![[r#"
            22..26 'self': &Self
            76..78 'f1': fn()
            86..88 'f2': fn(usize) -> u8
            107..109 'f3': fn(u8, u8) -> &u8
            130..178 '{     ...t(); }': ()
            136..138 'f1': fn()
            136..145 'f1.test()': bool
            151..153 'f2': fn(usize) -> u8
            151..160 'f2.test()': bool
            166..168 'f3': fn(u8, u8) -> &u8
            166..175 'f3.test()': bool
        "#]],
    );
}

#[test]
fn builtin_sized() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
trait Test { fn test(&self) -> bool; }
impl<T: Sized> Test for T {}

fn test() {
    1u8.test();
    (*"foo").test(); // not Sized
    (1u8, 1u8).test();
    (1u8, *"foo").test(); // not Sized
}"#,
        expect![[r#"
            22..26 'self': &Self
            79..194 '{     ...ized }': ()
            85..88 '1u8': u8
            85..95 '1u8.test()': bool
            101..116 '(*"foo").test()': {unknown}
            102..108 '*"foo"': str
            103..108 '"foo"': &str
            135..145 '(1u8, 1u8)': (u8, u8)
            135..152 '(1u8, ...test()': bool
            136..139 '1u8': u8
            141..144 '1u8': u8
            158..171 '(1u8, *"foo")': (u8, str)
            158..178 '(1u8, ...test()': {unknown}
            159..162 '1u8': u8
            164..170 '*"foo"': str
            165..170 '"foo"': &str
        "#]],
    );
}

#[test]
fn integer_range_iterate() {
    check_types(
        r#"
//- minicore: range, iterator
//- /main.rs crate:main
fn test() {
    for x in 0..100 { x; }
}                   //^ i32

trait Step {}
impl Step for i32 {}
impl Step for i64 {}

impl<A: Step> core::iter::Iterator for core::ops::Range<A> {
    type Item = A;
}
"#,
    );
}

#[test]
fn infer_closure_arg() {
    check_infer(
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
}"#,
        expect![[r#"
            52..126 '{     ...)(s) }': ()
            62..63 's': Option<i32>
            66..78 'Option::None': Option<i32>
            88..89 'f': |Option<i32>| -> ()
            92..111 '|x: Op...2>| {}': |Option<i32>| -> ()
            93..94 'x': Option<i32>
            109..111 '{}': ()
            117..124 '(&f)(s)': ()
            118..120 '&f': &|Option<i32>| -> ()
            119..120 'f': |Option<i32>| -> ()
            122..123 's': Option<i32>
        "#]],
    );
}

#[test]
fn dyn_fn_param_informs_call_site_closure_signature() {
    cov_mark::check!(dyn_fn_param_informs_call_site_closure_signature);
    check_types(
        r#"
//- minicore: fn, coerce_unsized
struct S;
impl S {
    fn inherent(&self) -> u8 { 0 }
}
fn take_dyn_fn(f: &dyn Fn(S)) {}

fn f() {
    take_dyn_fn(&|x| { x.inherent(); });
                     //^^^^^^^^^^^^ u8
}
        "#,
    );
}

#[test]
fn infer_fn_trait_arg() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn, option
fn foo<F, T>(f: F) -> T
where
    F: Fn(Option<i32>) -> T,
{
    let s = None;
    f(s)
}
"#,
        expect![[r#"
            13..14 'f': F
            59..89 '{     ...f(s) }': T
            69..70 's': Option<i32>
            73..77 'None': Option<i32>
            83..84 'f': F
            83..87 'f(s)': T
            85..86 's': Option<i32>
        "#]],
    );
}

#[test]
fn infer_box_fn_arg() {
    // The type mismatch is because we don't define Unsize and CoerceUnsized
    check_infer_with_mismatches(
        r#"
//- minicore: fn, deref, option
#[lang = "owned_box"]
pub struct Box<T: ?Sized> {
    inner: *mut T,
}

impl<T: ?Sized> core::ops::Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.inner
    }
}

fn foo() {
    let s = None;
    let f: Box<dyn FnOnce(&Option<i32>)> = box (|ps| {});
    f(&s);
}"#,
        expect![[r#"
            154..158 'self': &Box<T>
            166..193 '{     ...     }': &T
            176..187 '&self.inner': &*mut T
            177..181 'self': &Box<T>
            177..187 'self.inner': *mut T
            206..296 '{     ...&s); }': ()
            216..217 's': Option<i32>
            220..224 'None': Option<i32>
            234..235 'f': Box<dyn FnOnce(&Option<i32>)>
            269..282 'box (|ps| {})': Box<|&Option<i32>| -> ()>
            274..281 '|ps| {}': |&Option<i32>| -> ()
            275..277 'ps': &Option<i32>
            279..281 '{}': ()
            288..289 'f': Box<dyn FnOnce(&Option<i32>)>
            288..293 'f(&s)': ()
            290..292 '&s': &Option<i32>
            291..292 's': Option<i32>
            269..282: expected Box<dyn FnOnce(&Option<i32>)>, got Box<|&Option<i32>| -> ()>
        "#]],
    );
}

#[test]
fn infer_dyn_fn_output() {
    check_types(
        r#"
//- minicore: fn
fn foo() {
    let f: &dyn Fn() -> i32;
    f();
  //^^^ i32
}"#,
    );
}

#[test]
fn infer_dyn_fn_once_output() {
    check_types(
        r#"
//- minicore: fn
fn foo() {
    let f: dyn FnOnce() -> i32;
    f();
  //^^^ i32
}"#,
    );
}

#[test]
fn variable_kinds_1() {
    check_types(
        r#"
trait Trait<T> { fn get(self, t: T) -> T; }
struct S;
impl Trait<u128> for S {}
impl Trait<f32> for S {}
fn test() {
    S.get(1);
  //^^^^^^^^ u128
    S.get(1.);
  //^^^^^^^^^ f32
}
        "#,
    );
}

#[test]
fn variable_kinds_2() {
    check_types(
        r#"
trait Trait { fn get(self) -> Self; }
impl Trait for u128 {}
impl Trait for f32 {}
fn test() {
    1.get();
  //^^^^^^^ u128
    (1.).get();
  //^^^^^^^^^^ f32
}
        "#,
    );
}

#[test]
fn underscore_import() {
    check_types(
        r#"
mod tr {
    pub trait Tr {
        fn method(&self) -> u8 { 0 }
    }
}

struct Tr;
impl crate::tr::Tr for Tr {}

use crate::tr::Tr as _;
fn test() {
    Tr.method();
  //^^^^^^^^^^^ u8
}
    "#,
    );
}

#[test]
fn inner_use() {
    check_types(
        r#"
mod m {
    pub trait Tr {
        fn method(&self) -> u8 { 0 }
    }

    impl Tr for () {}
}

fn f() {
    use m::Tr;

    ().method();
  //^^^^^^^^^^^ u8
}
        "#,
    );
}

#[test]
fn trait_in_scope_with_inner_item() {
    check_infer(
        r#"
mod m {
    pub trait Tr {
        fn method(&self) -> u8 { 0 }
    }

    impl Tr for () {}
}

use m::Tr;

fn f() {
    fn inner() {
        ().method();
      //^^^^^^^^^^^ u8
    }
}"#,
        expect![[r#"
            46..50 'self': &Self
            58..63 '{ 0 }': u8
            60..61 '0': u8
            115..185 '{     ...   } }': ()
            132..183 '{     ...     }': ()
            142..144 '()': ()
            142..153 '().method()': u8
        "#]],
    );
}

#[test]
fn inner_use_in_block() {
    check_types(
        r#"
mod m {
    pub trait Tr {
        fn method(&self) -> u8 { 0 }
    }

    impl Tr for () {}
}

fn f() {
    {
        use m::Tr;

        ().method();
      //^^^^^^^^^^^ u8
    }

    {
        ().method();
      //^^^^^^^^^^^ {unknown}
    }

    ().method();
  //^^^^^^^^^^^ {unknown}
}
        "#,
    );
}

#[test]
fn nested_inner_function_calling_self() {
    check_infer(
        r#"
struct S;
fn f() {
    fn inner() -> S {
        let s = inner();
    }
}"#,
        expect![[r#"
            17..73 '{     ...   } }': ()
            39..71 '{     ...     }': S
            53..54 's': S
            57..62 'inner': fn inner() -> S
            57..64 'inner()': S
        "#]],
    )
}

#[test]
fn infer_default_trait_type_parameter() {
    check_infer(
        r#"
struct A;

trait Op<RHS=Self> {
    type Output;

    fn do_op(self, rhs: RHS) -> Self::Output;
}

impl Op for A {
    type Output = bool;

    fn do_op(self, rhs: Self) -> Self::Output {
        true
    }
}

fn test() {
    let x = A;
    let y = A;
    let r = x.do_op(y);
}"#,
        expect![[r#"
            63..67 'self': Self
            69..72 'rhs': RHS
            153..157 'self': A
            159..162 'rhs': A
            186..206 '{     ...     }': bool
            196..200 'true': bool
            220..277 '{     ...(y); }': ()
            230..231 'x': A
            234..235 'A': A
            245..246 'y': A
            249..250 'A': A
            260..261 'r': bool
            264..265 'x': A
            264..274 'x.do_op(y)': bool
            272..273 'y': A
        "#]],
    )
}

#[test]
fn qualified_path_as_qualified_trait() {
    check_infer(
        r#"
mod foo {

    pub trait Foo {
        type Target;
    }
    pub trait Bar {
        type Output;
        fn boo() -> Self::Output {
            loop {}
        }
    }
}

struct F;
impl foo::Foo for F {
    type Target = ();
}
impl foo::Bar for F {
    type Output = <F as foo::Foo>::Target;
}

fn foo() {
    use foo::Bar;
    let x = <F as Bar>::boo();
}"#,
        expect![[r#"
            132..163 '{     ...     }': Bar::Output<Self>
            146..153 'loop {}': !
            151..153 '{}': ()
            306..358 '{     ...o(); }': ()
            334..335 'x': ()
            338..353 '<F as Bar>::boo': fn boo<F>() -> <F as Bar>::Output
            338..355 '<F as ...:boo()': ()
        "#]],
    );
}

#[test]
fn renamed_extern_crate_in_block() {
    check_types(
        r#"
//- /lib.rs crate:lib deps:serde
use serde::Deserialize;

struct Foo {}

const _ : () = {
    extern crate serde as _serde;
    impl _serde::Deserialize for Foo {
        fn deserialize() -> u8 { 0 }
    }
};

fn foo() {
    Foo::deserialize();
  //^^^^^^^^^^^^^^^^^^ u8
}

//- /serde.rs crate:serde

pub trait Deserialize {
    fn deserialize() -> u8;
}"#,
    );
}

#[test]
fn bin_op_with_rhs_is_self_for_assoc_bound() {
    check_no_mismatches(
        r#"//- minicore: eq
        fn repro<T>(t: T) -> bool
where
    T: Request,
    T::Output: Convertable,
{
    let a = execute(&t).convert();
    let b = execute(&t).convert();
    a.eq(&b);
    let a = execute(&t).convert2();
    let b = execute(&t).convert2();
    a.eq(&b)
}
fn execute<T>(t: &T) -> T::Output
where
    T: Request,
{
    <T as Request>::output()
}
trait Convertable {
    type TraitSelf: PartialEq<Self::TraitSelf>;
    type AssocAsDefaultSelf: PartialEq;
    fn convert(self) -> Self::AssocAsDefaultSelf;
    fn convert2(self) -> Self::TraitSelf;
}
trait Request {
    type Output;
    fn output() -> Self::Output;
}
     "#,
    );
}

#[test]
fn bin_op_adt_with_rhs_primitive() {
    check_infer_with_mismatches(
        r#"
//- minicore: add
struct Wrapper(u32);
impl core::ops::Add<u32> for Wrapper {
    type Output = Self;
    fn add(self, rhs: u32) -> Wrapper {
        Wrapper(rhs)
    }
}
fn main(){
    let wrapped = Wrapper(10);
    let num: u32 = 2;
    let res = wrapped + num;

}"#,
        expect![[r#"
            95..99 'self': Wrapper
            101..104 'rhs': u32
            122..150 '{     ...     }': Wrapper
            132..139 'Wrapper': Wrapper(u32) -> Wrapper
            132..144 'Wrapper(rhs)': Wrapper
            140..143 'rhs': u32
            162..248 '{     ...um;  }': ()
            172..179 'wrapped': Wrapper
            182..189 'Wrapper': Wrapper(u32) -> Wrapper
            182..193 'Wrapper(10)': Wrapper
            190..192 '10': u32
            203..206 'num': u32
            214..215 '2': u32
            225..228 'res': Wrapper
            231..238 'wrapped': Wrapper
            231..244 'wrapped + num': Wrapper
            241..244 'num': u32
        "#]],
    )
}

#[test]
fn builtin_binop_expectation_works_on_single_reference() {
    check_types(
        r#"
//- minicore: add
use core::ops::Add;
impl Add<i32> for i32 { type Output = i32 }
impl Add<&i32> for i32 { type Output = i32 }
impl Add<u32> for u32 { type Output = u32 }
impl Add<&u32> for u32 { type Output = u32 }

struct V<T>;
impl<T> V<T> {
    fn default() -> Self { loop {} }
    fn get(&self, _: &T) -> &T { loop {} }
}

fn take_u32(_: u32) {}
fn minimized() {
    let v = V::default();
    let p = v.get(&0);
      //^ &u32
    take_u32(42 + p);
}
"#,
    );
}

#[test]
fn no_builtin_binop_expectation_for_general_ty_var() {
    // FIXME: Ideally type mismatch should be reported on `take_u32(42 - p)`.
    check_types(
        r#"
//- minicore: add
use core::ops::Add;
impl Add<i32> for i32 { type Output = i32; }
impl Add<&i32> for i32 { type Output = i32; }
// This is needed to prevent chalk from giving unique solution to `i32: Add<&?0>` after applying
// fallback to integer type variable for `42`.
impl Add<&()> for i32 { type Output = (); }

struct V<T>;
impl<T> V<T> {
    fn default() -> Self { loop {} }
    fn get(&self) -> &T { loop {} }
}

fn take_u32(_: u32) {}
fn minimized() {
    let v = V::default();
    let p = v.get();
      //^ &{unknown}
    take_u32(42 + p);
}
"#,
    );
}

#[test]
fn no_builtin_binop_expectation_for_non_builtin_types() {
    check_no_mismatches(
        r#"
//- minicore: default, eq
struct S;
impl Default for S { fn default() -> Self { S } }
impl Default for i32 { fn default() -> Self { 0 } }
impl PartialEq<S> for i32 { fn eq(&self, _: &S) -> bool { true } }
impl PartialEq<i32> for i32 { fn eq(&self, _: &S) -> bool { true } }

fn take_s(_: S) {}
fn test() {
    let s = Default::default();
    let _eq = 0 == s;
    take_s(s);
}
"#,
    )
}

#[test]
fn array_length() {
    check_infer(
        r#"
trait T {
    type Output;
    fn do_thing(&self) -> Self::Output;
}

impl T for [u8; 4] {
    type Output = usize;
    fn do_thing(&self) -> Self::Output {
        2
    }
}

impl T for [u8; 2] {
    type Output = u8;
    fn do_thing(&self) -> Self::Output {
        2
    }
}

fn main() {
    let v = [0u8; 2];
    let v2 = v.do_thing();
    let v3 = [0u8; 4];
    let v4 = v3.do_thing();
}
"#,
        expect![[r#"
            44..48 'self': &Self
            133..137 'self': &[u8; 4]
            155..172 '{     ...     }': usize
            165..166 '2': usize
            236..240 'self': &[u8; 2]
            258..275 '{     ...     }': u8
            268..269 '2': u8
            289..392 '{     ...g(); }': ()
            299..300 'v': [u8; 2]
            303..311 '[0u8; 2]': [u8; 2]
            304..307 '0u8': u8
            309..310 '2': usize
            321..323 'v2': u8
            326..327 'v': [u8; 2]
            326..338 'v.do_thing()': u8
            348..350 'v3': [u8; 4]
            353..361 '[0u8; 4]': [u8; 4]
            354..357 '0u8': u8
            359..360 '4': usize
            371..373 'v4': usize
            376..378 'v3': [u8; 4]
            376..389 'v3.do_thing()': usize
        "#]],
    )
}

#[test]
fn const_generics() {
    check_infer(
        r#"
trait T {
    type Output;
    fn do_thing(&self) -> Self::Output;
}

impl<const L: usize> T for [u8; L] {
    type Output = [u8; L];
    fn do_thing(&self) -> Self::Output {
        *self
    }
}

fn main() {
    let v = [0u8; 2];
    let v2 = v.do_thing();
}
"#,
        expect![[r#"
            44..48 'self': &Self
            151..155 'self': &[u8; L]
            173..194 '{     ...     }': [u8; L]
            183..188 '*self': [u8; L]
            184..188 'self': &[u8; L]
            208..260 '{     ...g(); }': ()
            218..219 'v': [u8; 2]
            222..230 '[0u8; 2]': [u8; 2]
            223..226 '0u8': u8
            228..229 '2': usize
            240..242 'v2': [u8; 2]
            245..246 'v': [u8; 2]
            245..257 'v.do_thing()': [u8; 2]
        "#]],
    )
}

#[test]
fn fn_returning_unit() {
    check_infer_with_mismatches(
        r#"
//- minicore: fn
fn test<F: FnOnce()>(f: F) {
    let _: () = f();
}"#,
        expect![[r#"
            21..22 'f': F
            27..51 '{     ...f(); }': ()
            37..38 '_': ()
            45..46 'f': F
            45..48 'f()': ()
        "#]],
    );
}

#[test]
fn trait_in_scope_of_trait_impl() {
    check_infer(
        r#"
mod foo {
    pub trait Foo {
        fn foo(self);
        fn bar(self) -> usize { 0 }
    }
}
impl foo::Foo for u32 {
    fn foo(self) {
        let _x = self.bar();
    }
}
    "#,
        expect![[r#"
            45..49 'self': Self
            67..71 'self': Self
            82..87 '{ 0 }': usize
            84..85 '0': usize
            131..135 'self': u32
            137..173 '{     ...     }': ()
            151..153 '_x': usize
            156..160 'self': u32
            156..166 'self.bar()': usize
        "#]],
    );
}

#[test]
fn infer_async_ret_type() {
    check_types(
        r#"
//- minicore: future, result
struct Fooey;

impl Fooey {
    fn collect<B: Convert>(self) -> B {
        B::new()
    }
}

trait Convert {
    fn new() -> Self;
}
impl Convert for u32 {
    fn new() -> Self { 0 }
}

async fn get_accounts() -> Result<u32, ()> {
    let ret = Fooey.collect();
    //        ^^^^^^^^^^^^^^^ u32
    Ok(ret)
}
"#,
    );
}

#[test]
fn local_impl_1() {
    check!(block_local_impls);
    check_types(
        r#"
trait Trait<T> {
    fn foo(&self) -> T;
}

fn test() {
    struct S;
    impl Trait<u32> for S {
        fn foo(&self) -> u32 { 0 }
    }

    S.foo();
 // ^^^^^^^ u32
}
"#,
    );
}

#[test]
fn local_impl_2() {
    check!(block_local_impls);
    check_types(
        r#"
struct S;

fn test() {
    trait Trait<T> {
        fn foo(&self) -> T;
    }
    impl Trait<u32> for S {
        fn foo(&self) -> u32 { 0 }
    }

    S.foo();
 // ^^^^^^^ u32
}
"#,
    );
}

#[test]
fn local_impl_3() {
    check!(block_local_impls);
    check_types(
        r#"
trait Trait<T> {
    fn foo(&self) -> T;
}

fn test() {
    struct S1;
    {
        struct S2;

        impl Trait<S1> for S2 {
            fn foo(&self) -> S1 { S1 }
        }

        S2.foo();
     // ^^^^^^^^ S1
    }
}
"#,
    );
}

#[test]
fn associated_type_sized_bounds() {
    check_infer(
        r#"
//- minicore: sized
struct Yes;
trait IsSized { const IS_SIZED: Yes; }
impl<T: Sized> IsSized for T { const IS_SIZED: Yes = Yes; }

trait Foo {
    type Explicit: Sized;
    type Implicit;
    type Relaxed: ?Sized;
}
fn f<F: Foo>() {
    F::Explicit::IS_SIZED;
    F::Implicit::IS_SIZED;
    F::Relaxed::IS_SIZED;
}
"#,
        expect![[r#"
            104..107 'Yes': Yes
            212..295 '{     ...ZED; }': ()
            218..239 'F::Exp..._SIZED': Yes
            245..266 'F::Imp..._SIZED': Yes
            272..292 'F::Rel..._SIZED': {unknown}
        "#]],
    );
}

#[test]
fn dyn_map() {
    check_types(
        r#"
pub struct Key<K, V, P = (K, V)> {}

pub trait Policy {
    type K;
    type V;
}

impl<K, V> Policy for (K, V) {
    type K = K;
    type V = V;
}

pub struct KeyMap<KEY> {}

impl<P: Policy> KeyMap<Key<P::K, P::V, P>> {
    pub fn get(&self, key: &P::K) -> P::V {
        loop {}
    }
}

struct Fn {}
struct FunctionId {}

fn test() {
    let key_map: &KeyMap<Key<Fn, FunctionId>> = loop {};
    let key;
    let result = key_map.get(key);
      //^^^^^^ FunctionId
}
"#,
    )
}

#[test]
fn dyn_multiple_auto_traits_in_different_order() {
    check_no_mismatches(
        r#"
auto trait Send {}
auto trait Sync {}

fn f(t: &(dyn Sync + Send)) {}
fn g(t: &(dyn Send + Sync)) {
    f(t);
}
        "#,
    );

    check_no_mismatches(
        r#"
auto trait Send {}
auto trait Sync {}
trait T {}

fn f(t: &(dyn T + Send + Sync)) {}
fn g(t: &(dyn Sync + T + Send)) {
    f(t);
}
        "#,
    );

    check_infer_with_mismatches(
        r#"
auto trait Send {}
auto trait Sync {}
trait T1 {}
trait T2 {}

fn f(t: &(dyn T1 + T2 + Send + Sync)) {}
fn g(t: &(dyn Sync + T2 + T1 + Send)) {
    f(t);
}
        "#,
        expect![[r#"
            68..69 't': &{unknown}
            101..103 '{}': ()
            109..110 't': &{unknown}
            142..155 '{     f(t); }': ()
            148..149 'f': fn f(&{unknown})
            148..152 'f(t)': ()
            150..151 't': &{unknown}
        "#]],
    );

    check_no_mismatches(
        r#"
auto trait Send {}
auto trait Sync {}
trait T {
    type Proj: Send + Sync;
}

fn f(t: &(dyn T<Proj = ()>  + Send + Sync)) {}
fn g(t: &(dyn Sync + T<Proj = ()> + Send)) {
    f(t);
}
        "#,
    );
}

#[test]
fn dyn_multiple_projection_bounds() {
    check_no_mismatches(
        r#"
trait Trait {
    type T;
    type U;
}

fn f(t: &dyn Trait<T = (), U = ()>) {}
fn g(t: &dyn Trait<U = (), T = ()>) {
    f(t);
}
        "#,
    );

    check_types(
        r#"
trait Trait {
    type T;
}

fn f(t: &dyn Trait<T = (), T = ()>) {}
   //^&{unknown}
        "#,
    );
}

#[test]
fn dyn_duplicate_auto_trait() {
    check_no_mismatches(
        r#"
auto trait Send {}

fn f(t: &(dyn Send + Send)) {}
fn g(t: &(dyn Send)) {
    f(t);
}
        "#,
    );

    check_no_mismatches(
        r#"
auto trait Send {}
trait T {}

fn f(t: &(dyn T + Send + Send)) {}
fn g(t: &(dyn T + Send)) {
    f(t);
}
        "#,
    );
}

#[test]
fn gats_in_path() {
    check_types(
        r#"
//- minicore: deref
use core::ops::Deref;
trait PointerFamily {
    type Pointer<T>: Deref<Target = T>;
}

fn f<P: PointerFamily>(p: P::Pointer<i32>) {
    let a = *p;
      //^ i32
}
fn g<P: PointerFamily>(p: <P as PointerFamily>::Pointer<i32>) {
    let a = *p;
      //^ i32
}
        "#,
    );
}

#[test]
fn gats_with_impl_trait() {
    // FIXME: the last function (`fn i()`) is not valid Rust as of this writing because you cannot
    // specify the same associated type multiple times even if their arguments are different (c.f.
    // `fn h()`, which is valid). Reconsider how to treat these invalid types.
    check_types(
        r#"
//- minicore: deref
use core::ops::Deref;

trait Trait {
    type Assoc<T>: Deref<Target = T>;
    fn get<U>(&self) -> Self::Assoc<U>;
}

fn f<T>(v: impl Trait) {
    let a = v.get::<i32>().deref();
      //^ &i32
    let a = v.get::<T>().deref();
      //^ &T
}
fn g<'a, T: 'a>(v: impl Trait<Assoc<T> = &'a T>) {
    let a = v.get::<T>();
      //^ &T
    let a = v.get::<()>();
      //^ Trait::Assoc<(), impl Trait<Assoc<T> = &T>>
}
fn h<'a>(v: impl Trait<Assoc<i32> = &'a i32> + Trait<Assoc<i64> = &'a i64>) {
    let a = v.get::<i32>();
      //^ &i32
    let a = v.get::<i64>();
      //^ &i64
}
fn i<'a>(v: impl Trait<Assoc<i32> = &'a i32, Assoc<i64> = &'a i64>) {
    let a = v.get::<i32>();
      //^ &i32
    let a = v.get::<i64>();
      //^ &i64
}
    "#,
    );
}

#[test]
fn gats_with_dyn() {
    // This test is here to keep track of how we infer things despite traits with GATs being not
    // object-safe currently.
    // FIXME: reconsider how to treat these invalid types.
    check_infer_with_mismatches(
        r#"
//- minicore: deref
use core::ops::Deref;

trait Trait {
    type Assoc<T>: Deref<Target = T>;
    fn get<U>(&self) -> Self::Assoc<U>;
}

fn f<'a>(v: &dyn Trait<Assoc<i32> = &'a i32>) {
    v.get::<i32>().deref();
}
    "#,
        expect![[r#"
            90..94 'self': &Self
            127..128 'v': &(dyn Trait<Assoc<i32> = &i32>)
            164..195 '{     ...f(); }': ()
            170..171 'v': &(dyn Trait<Assoc<i32> = &i32>)
            170..184 'v.get::<i32>()': &i32
            170..192 'v.get:...eref()': &i32
        "#]],
    );
}

#[test]
fn gats_in_associated_type_binding() {
    check_types(
        r#"
trait Trait {
    type Assoc<T>;
    fn get<U>(&self) -> Self::Assoc<U>;
}

fn f<T>(t: T)
where
    T: Trait<Assoc<i32> = u32>,
    T: Trait<Assoc<isize> = usize>,
{
    let a = t.get::<i32>();
      //^ u32
    let a = t.get::<isize>();
      //^ usize
    let a = t.get::<()>();
      //^ Trait::Assoc<(), T>
}

    "#,
    );
}

#[test]
fn bin_op_with_scalar_fallback() {
    // Extra impls are significant so that chalk doesn't give us definite guidances.
    check_types(
        r#"
//- minicore: add
use core::ops::Add;

struct Vec2<T>(T, T);

impl Add for Vec2<i32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { loop {} }
}
impl Add for Vec2<u32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { loop {} }
}
impl Add for Vec2<f32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { loop {} }
}
impl Add for Vec2<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output { loop {} }
}

fn test() {
    let a = Vec2(1, 2);
    let b = Vec2(3, 4);
    let c = a + b;
      //^ Vec2<i32>
    let a = Vec2(1., 2.);
    let b = Vec2(3., 4.);
    let c = a + b;
      //^ Vec2<f64>
}
"#,
    );
}

#[test]
fn trait_method_with_scalar_fallback() {
    check_types(
        r#"
trait Trait {
    type Output;
    fn foo(&self) -> Self::Output;
}
impl<T> Trait for T {
    type Output = T;
    fn foo(&self) -> Self::Output { loop {} }
}
fn test() {
    let a = 42;
    let b = a.foo();
      //^ i32
    let a = 3.14;
    let b = a.foo();
      //^ f64
}
"#,
    );
}
