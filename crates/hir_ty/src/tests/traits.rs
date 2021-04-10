use expect_test::expect;

use super::{check_infer, check_infer_with_mismatches, check_types};

#[test]
fn infer_await() {
    check_types(
        r#"
//- /main.rs crate:main deps:core
struct IntFuture;

impl Future for IntFuture {
    type Output = u64;
}

fn test() {
    let r = IntFuture;
    let v = r.await;
    v;
} //^ u64

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
}

#[test]
fn infer_async() {
    check_types(
        r#"
//- /main.rs crate:main deps:core
async fn foo() -> u64 {
            128
}

fn test() {
    let r = foo();
    let v = r.await;
    v;
} //^ u64

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
}

#[test]
fn infer_desugar_async() {
    check_types(
        r#"
//- /main.rs crate:main deps:core
async fn foo() -> u64 {
            128
}

fn test() {
    let r = foo();
    r;
} //^ impl Future<Output = u64>

//- /core.rs crate:core
#[prelude_import] use future::*;
mod future {
    trait Future {
        type Output;
    }
}

"#,
    );
}

#[test]
fn infer_async_block() {
    check_types(
        r#"
//- /main.rs crate:main deps:core
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
        let y = Option::None;
        y
    //  ^ Option<u64>
    };
    let _: Option<u64> = c.await;
    c;
//  ^ impl Future<Output = Option<u64>>
}

enum Option<T> { None, Some(T) }

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
}

#[test]
fn infer_try() {
    check_types(
        r#"
//- /main.rs crate:main deps:core
fn test() {
    let r: Result<i32, u64> = Result::Ok(1);
    let v = r?;
    v;
} //^ i32

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
}

#[test]
fn infer_for_loop() {
    check_types(
        r#"
//- /main.rs crate:main deps:core,alloc
use alloc::collections::Vec;

fn test() {
    let v = Vec::new();
    v.push("foo");
    for x in v {
        x;
    } //^ &str
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
        pub fn new() -> Self { Vec {} }
        pub fn push(&mut self, t: T) { }
    }

    impl<T> IntoIterator for Vec<T> {
        type Item=T;
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
    check_infer(
        r#"
        trait Trait<T> {}
        struct S<T>(T);
        impl<U> Trait<U> for S<U> {}
        fn foo<T: Trait<u32>>(t: T) {}
        fn test() {
            let s = S(unknown);
            foo(s);
        }
        "#,
        expect![[r#"
            85..86 't': T
            91..93 '{}': ()
            104..143 '{     ...(s); }': ()
            114..115 's': S<u32>
            118..119 'S': S<u32>(u32) -> S<u32>
            118..128 'S(unknown)': S<u32>
            120..127 'unknown': u32
            134..137 'foo': fn foo<S<u32>>(S<u32>)
            134..140 'foo(s)': ()
            138..139 's': S<u32>
        "#]],
    );
}

#[test]
fn infer_from_bound_2() {
    check_infer(
        r#"
        trait Trait<T> {}
        struct S<T>(T);
        impl<U> Trait<U> for S<U> {}
        fn foo<U, T: Trait<U>>(t: T) -> U {}
        fn test() {
            let s = S(unknown);
            let x: u32 = foo(s);
        }
        "#,
        expect![[r#"
            86..87 't': T
            97..99 '{}': ()
            110..162 '{     ...(s); }': ()
            120..121 's': S<u32>
            124..125 'S': S<u32>(u32) -> S<u32>
            124..134 'S(unknown)': S<u32>
            126..133 'unknown': u32
            144..145 'x': u32
            153..156 'foo': fn foo<u32, S<u32>>(S<u32>) -> u32
            153..159 'foo(s)': u32
            157..158 's': S<u32>
        "#]],
    );
}

#[test]
fn trait_default_method_self_bound_implements_trait() {
    cov_mark::check!(trait_self_implements_self);
    check_infer(
        r#"
        trait Trait {
            fn foo(&self) -> i64;
            fn bar(&self) -> {
                let x = self.foo();
            }
        }
        "#,
        expect![[r#"
            26..30 'self': &Self
            52..56 'self': &Self
            61..96 '{     ...     }': ()
            75..76 'x': i64
            79..83 'self': &Self
            79..89 'self.foo()': i64
        "#]],
    );
}

#[test]
fn trait_default_method_self_bound_implements_super_trait() {
    check_infer(
        r#"
        trait SuperTrait {
            fn foo(&self) -> i64;
        }
        trait Trait: SuperTrait {
            fn bar(&self) -> {
                let x = self.foo();
            }
        }
        "#,
        expect![[r#"
            31..35 'self': &Self
            85..89 'self': &Self
            94..129 '{     ...     }': ()
            108..109 'x': i64
            112..116 'self': &Self
            112..122 'self.foo()': i64
        "#]],
    );
}

#[test]
fn infer_project_associated_type() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            108..261 '{     ...ter; }': ()
            118..119 'x': u32
            145..146 '1': u32
            156..157 'y': Iterable::Item<T>
            183..192 'no_matter': Iterable::Item<T>
            202..203 'z': Iterable::Item<T>
            215..224 'no_matter': Iterable::Item<T>
            234..235 'a': Iterable::Item<T>
            249..258 'no_matter': Iterable::Item<T>
        "#]],
    );
}

#[test]
fn infer_return_associated_type() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
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
        "#]],
    );
}

#[test]
fn infer_associated_type_bound() {
    check_infer(
        r#"
        trait Iterable {
            type Item;
        }
        fn test<T: Iterable<Item=u32>>() {
            let y: T::Item = unknown;
        }
        "#,
        expect![[r#"
            67..100 '{     ...own; }': ()
            77..78 'y': u32
            90..97 'unknown': u32
        "#]],
    );
}

#[test]
fn infer_const_body() {
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
        }
        "#,
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
        }
        "#,
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
            17..20 '[9]': [i32; _]
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
//- /main.rs crate:main deps:std
struct Bar;
struct Foo;

impl std::ops::Index<u32> for Bar {
    type Output = Foo;
}

fn test() {
    let a = Bar;
    let b = a[1u32];
    b;
} //^ Foo

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
}

#[test]
fn infer_ops_index_int() {
    check_types(
        r#"
//- /main.rs crate:main deps:std
struct Bar;
struct Foo;

impl std::ops::Index<u32> for Bar {
    type Output = Foo;
}

struct Range;
impl std::ops::Index<Range> for Bar {
    type Output = Bar;
}

fn test() {
    let a = Bar;
    let b = a[1];
    b;
  //^ Foo
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
}

#[test]
fn infer_ops_index_autoderef() {
    check_types(
        r#"
//- /main.rs crate:main deps:std
fn test() {
    let a = &[1u32, 2, 3];
    let b = a[1u32];
    b;
} //^ u32

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
}

#[test]
fn deref_trait() {
    check_types(
        r#"
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
    (*s, s.foo());
} //^ (S, u128)
"#,
    );
}

#[test]
fn deref_trait_with_inference_var() {
    check_types(
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
    let b = (*a);
          //^ S
    foo(a);
}
"#,
    );
}

#[test]
fn deref_trait_infinite_recursion() {
    check_types(
        r#"
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
    s.foo();
}       //^ {unknown}
"#,
    );
}

#[test]
fn deref_trait_with_question_mark_size() {
    check_types(
        r#"
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
    (*s, s.foo());
} //^ (S, u128)
"#,
    );
}

#[test]
fn obligation_from_function_clause() {
    check_types(
        r#"
struct S;

trait Trait<T> {}
impl Trait<u32> for S {}

fn foo<T: Trait<U>, U>(t: T) -> U {}

fn test(s: S) {
    (foo(s));
} //^ u32
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
    fn foo<T: Trait<U>, U>(&self, t: T) -> U {}
}

fn test() {
    O.foo(S);
}      //^ isize
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
    fn foo<U>(&self) -> U where Self: Trait<U> {}
}

fn test() {
    S.foo();
}       //^ i64
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
    fn foo(&self) -> U {}
}

fn test(o: O<S>) {
    o.foo();
}       //^ &str
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
                             //^ u128
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
                       //^ {unknown}
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
                              //^ u128
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
                       //^ {unknown}
"#,
    );
}

#[test]
fn generic_param_env_deref() {
    check_types(
        r#"
#[lang = "deref"]
trait Deref {
    type Target;
}
trait Trait {}
impl<T> Deref for T where T: Trait {
    type Target = i128;
}
fn test<T: Trait>(t: T) { (*t); }
                        //^ i128
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
        "#,
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
        "#,
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
        "#,
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
        trait Trait {}
        fn foo(x: impl Trait) { loop {} }
        struct S;
        impl Trait for S {}

        fn test() {
            let f: fn(S) -> () = foo;
        }
        "#,
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
        "#,
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
        trait Trait<T> {
            fn foo(&self) -> T;
        }
        fn bar() -> impl Trait<u64> { loop {} }

        fn test() {
            let a = bar();
            a.foo();
        }
        "#,
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
        "#,
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
fn dyn_trait() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            29..33 'self': &Self
            54..58 'self': &Self
            97..99 '{}': ()
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
        "#,
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
        "#,
        expect![[r#"
            26..30 'self': &Self
            60..62 '{}': ()
            72..73 'x': dyn Trait
            82..83 'y': &dyn Trait
            100..175 '{     ...o(); }': ()
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
}

#[test]
fn weird_bounds() {
    check_infer(
        r#"
        trait Trait {}
        fn test(a: impl Trait + 'lifetime, b: impl 'lifetime, c: impl (Trait), d: impl ('lifetime), e: impl ?Sized, f: impl Trait + ?Sized) {}
        "#,
        expect![[r#"
            23..24 'a': impl Trait
            50..51 'b': impl
            69..70 'c': impl Trait
            86..87 'd': impl
            107..108 'e': impl
            123..124 'f': impl Trait
            147..149 '{}': ()
        "#]],
    );
}

#[test]
#[ignore]
fn error_bound_chalk() {
    check_types(
        r#"
trait Trait {
    fn foo(&self) -> u32 {}
}

fn test(x: (impl Trait + UnknownTrait)) {
    x.foo();
}       //^ u32
"#,
    );
}

#[test]
fn assoc_type_bindings() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            49..50 't': T
            77..79 '{}': ()
            111..112 't': T
            122..124 '{}': ()
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
        node.clone();
    }            //^ {unknown}
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
        }
        "#,
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
    trait Trait {
        fn foo(&self) -> u32 {}
    }
}

fn test<T: foo::Trait>(x: T) {
    x.foo();
}      //^ u32
"#,
    );
}

#[test]
fn super_trait_method_resolution() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            49..53 'self': &Self
            62..64 '{}': ()
            181..182 'x': T
            187..188 'y': U
            193..222 '{     ...o(); }': ()
            199..200 'x': T
            199..206 'x.foo()': u32
            212..213 'y': U
            212..219 'y.foo()': u32
        "#]],
    );
}

#[test]
fn super_trait_impl_trait_method_resolution() {
    check_infer(
        r#"
        mod foo {
            trait SuperTrait {
                fn foo(&self) -> u32 {}
            }
        }
        trait Trait1: foo::SuperTrait {}

        fn test(x: &impl Trait1) {
            x.foo();
        }
        "#,
        expect![[r#"
            49..53 'self': &Self
            62..64 '{}': ()
            115..116 'x': &impl Trait1
            132..148 '{     ...o(); }': ()
            138..139 'x': &impl Trait1
            138..145 'x.foo()': u32
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
        }
        "#,
        expect![[r#"
            102..103 't': T
            113..115 '{}': ()
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
        trait FnOnce<Args> {
            type Output;

            fn call_once(self, args: Args) -> <Self as FnOnce<Args>>::Output;
        }

        fn test<F: FnOnce(u32, u64) -> u128>(f: F) {
            f.call_once((1, 2));
        }
        "#,
        expect![[r#"
            56..60 'self': Self
            62..66 'args': Args
            149..150 'f': F
            155..183 '{     ...2)); }': ()
            161..162 'f': F
            161..180 'f.call...1, 2))': u128
            173..179 '(1, 2)': (u32, u64)
            174..175 '1': u32
            177..178 '2': u64
        "#]],
    );
}

#[test]
fn fn_ptr_and_item() {
    check_infer_with_mismatches(
        r#"
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
        }
        "#,
        expect![[r#"
            74..78 'self': Self
            80..84 'args': Args
            139..143 'self': &Self
            243..247 'self': &Bar<F>
            260..271 '{ loop {} }': (A1, R)
            262..269 'loop {}': !
            267..269 '{}': ()
            355..359 'self': Opt<T>
            361..362 'f': F
            377..388 '{ loop {} }': Opt<U>
            379..386 'loop {}': !
            384..386 '{}': ()
            402..518 '{     ...(f); }': ()
            412..415 'bar': Bar<fn(u8) -> u32>
            441..444 'bar': Bar<fn(u8) -> u32>
            441..450 'bar.foo()': (u8, u32)
            461..464 'opt': Opt<u8>
            483..484 'f': fn(u8) -> u32
            505..508 'opt': Opt<u8>
            505..515 'opt.map(f)': Opt<u32>
            513..514 'f': fn(u8) -> u32
        "#]],
    );
}

#[test]
fn fn_trait_deref_with_ty_default() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            64..68 'self': &Self
            165..169 'self': Self
            171..175 'args': Args
            239..243 'self': &Foo
            254..256 '{}': ()
            334..335 'f': F
            354..356 '{}': ()
            443..689 '{     ...o(); }': ()
            453..458 'lazy1': Lazy<Foo, || -> Foo>
            475..484 'Lazy::new': fn new<Foo, || -> Foo>(|| -> Foo) -> Lazy<Foo, || -> Foo>
            475..492 'Lazy::...| Foo)': Lazy<Foo, || -> Foo>
            485..491 '|| Foo': || -> Foo
            488..491 'Foo': Foo
            502..504 'r1': usize
            507..512 'lazy1': Lazy<Foo, || -> Foo>
            507..518 'lazy1.foo()': usize
            560..575 'make_foo_fn_ptr': fn() -> Foo
            591..602 'make_foo_fn': fn make_foo_fn() -> Foo
            612..617 'lazy2': Lazy<Foo, fn() -> Foo>
            634..643 'Lazy::new': fn new<Foo, fn() -> Foo>(fn() -> Foo) -> Lazy<Foo, fn() -> Foo>
            634..660 'Lazy::...n_ptr)': Lazy<Foo, fn() -> Foo>
            644..659 'make_foo_fn_ptr': fn() -> Foo
            670..672 'r2': usize
            675..680 'lazy2': Lazy<Foo, fn() -> Foo>
            675..686 'lazy2.foo()': usize
            549..551 '{}': ()
        "#]],
    );
}

#[test]
fn closure_1() {
    check_infer_with_mismatches(
        r#"
        #[lang = "fn_once"]
        trait FnOnce<Args> {
            type Output;
        }

        enum Option<T> { Some(T), None }
        impl<T> Option<T> {
            fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Option<U> { loop {} }
        }

        fn test() {
            let x = Option::Some(1u32);
            x.map(|v| v + 1);
            x.map(|_v| 1u64);
            let y: Option<i64> = x.map(|_v| 1);
        }
        "#,
        expect![[r#"
            147..151 'self': Option<T>
            153..154 'f': F
            172..183 '{ loop {} }': Option<U>
            174..181 'loop {}': !
            179..181 '{}': ()
            197..316 '{     ... 1); }': ()
            207..208 'x': Option<u32>
            211..223 'Option::Some': Some<u32>(u32) -> Option<u32>
            211..229 'Option...(1u32)': Option<u32>
            224..228 '1u32': u32
            235..236 'x': Option<u32>
            235..251 'x.map(...v + 1)': Option<u32>
            241..250 '|v| v + 1': |u32| -> u32
            242..243 'v': u32
            245..246 'v': u32
            245..250 'v + 1': u32
            249..250 '1': u32
            257..258 'x': Option<u32>
            257..273 'x.map(... 1u64)': Option<u64>
            263..272 '|_v| 1u64': |u32| -> u64
            264..266 '_v': u32
            268..272 '1u64': u64
            283..284 'y': Option<i64>
            300..301 'x': Option<u32>
            300..313 'x.map(|_v| 1)': Option<i64>
            306..312 '|_v| 1': |u32| -> i64
            307..309 '_v': u32
            311..312 '1': i64
        "#]],
    );
}

#[test]
fn closure_2() {
    check_infer_with_mismatches(
        r#"
        #[lang = "add"]
        pub trait Add<Rhs = Self> {
            type Output;
            fn add(self, rhs: Rhs) -> Self::Output;
        }

        trait FnOnce<Args> {
            type Output;
        }

        impl Add for u64 {
            type Output = Self;
            fn add(self, rhs: u64) -> Self::Output {0}
        }

        impl Add for u128 {
            type Output = Self;
            fn add(self, rhs: u128) -> Self::Output {0}
        }

        fn test<F: FnOnce(u32) -> u64>(f: F) {
            f(1);
            let g = |v| v + 1;
            g(1u64);
            let h = |v| 1u128 + v;
        }
        "#,
        expect![[r#"
            72..76 'self': Self
            78..81 'rhs': Rhs
            203..207 'self': u64
            209..212 'rhs': u64
            235..238 '{0}': u64
            236..237 '0': u64
            297..301 'self': u128
            303..306 'rhs': u128
            330..333 '{0}': u128
            331..332 '0': u128
            368..369 'f': F
            374..450 '{     ...+ v; }': ()
            380..381 'f': F
            380..384 'f(1)': {unknown}
            382..383 '1': i32
            394..395 'g': |u64| -> u64
            398..407 '|v| v + 1': |u64| -> u64
            399..400 'v': u64
            402..403 'v': u64
            402..407 'v + 1': u64
            406..407 '1': u64
            413..414 'g': |u64| -> u64
            413..420 'g(1u64)': u64
            415..419 '1u64': u64
            430..431 'h': |u128| -> u128
            434..447 '|v| 1u128 + v': |u128| -> u128
            435..436 'v': u128
            438..443 '1u128': u128
            438..447 '1u128 + v': u128
            446..447 'v': u128
        "#]],
    );
}

#[test]
fn closure_as_argument_inference_order() {
    check_infer_with_mismatches(
        r#"
        #[lang = "fn_once"]
        trait FnOnce<Args> {
            type Output;
        }

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
        }
        "#,
        expect![[r#"
            94..95 'x': T
            100..101 'f': F
            111..122 '{ loop {} }': U
            113..120 'loop {}': !
            118..120 '{}': ()
            156..157 'f': F
            162..163 'x': T
            173..184 '{ loop {} }': U
            175..182 'loop {}': !
            180..182 '{}': ()
            219..223 'self': S
            271..275 'self': S
            277..278 'x': T
            283..284 'f': F
            294..305 '{ loop {} }': U
            296..303 'loop {}': !
            301..303 '{}': ()
            343..347 'self': S
            349..350 'f': F
            355..356 'x': T
            366..377 '{ loop {} }': U
            368..375 'loop {}': !
            373..375 '{}': ()
            391..550 '{     ... S); }': ()
            401..403 'x1': u64
            406..410 'foo1': fn foo1<S, u64, |S| -> u64>(S, |S| -> u64) -> u64
            406..429 'foo1(S...hod())': u64
            411..412 'S': S
            414..428 '|s| s.method()': |S| -> u64
            415..416 's': S
            418..419 's': S
            418..428 's.method()': u64
            439..441 'x2': u64
            444..448 'foo2': fn foo2<S, u64, |S| -> u64>(|S| -> u64, S) -> u64
            444..467 'foo2(|...(), S)': u64
            449..463 '|s| s.method()': |S| -> u64
            450..451 's': S
            453..454 's': S
            453..463 's.method()': u64
            465..466 'S': S
            477..479 'x3': u64
            482..483 'S': S
            482..507 'S.foo1...hod())': u64
            489..490 'S': S
            492..506 '|s| s.method()': |S| -> u64
            493..494 's': S
            496..497 's': S
            496..506 's.method()': u64
            517..519 'x4': u64
            522..523 'S': S
            522..547 'S.foo2...(), S)': u64
            529..543 '|s| s.method()': |S| -> u64
            530..531 's': S
            533..534 's': S
            533..543 's.method()': u64
            545..546 'S': S
        "#]],
    );
}

#[test]
fn fn_item_fn_trait() {
    check_types(
        r#"
#[lang = "fn_once"]
trait FnOnce<Args> {
    type Output;
}

struct S;

fn foo() -> S {}

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
}       //^ u32
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
}       //^ u32
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
        }
        "#,
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
}     //^ u32
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
}     //^ Ty<I>
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
}       //^ ()
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
}       //^ {unknown}
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_cycle_1() {
    // this is a legitimate cycle
    check_types(
        r#"
trait Trait {
    type Item;
}

trait Trait2<T> {}

fn test<T: Trait>() where T: Trait2<T::Item> {
    let x: T::Item = no_matter;
}                       //^ {unknown}
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
}                   //^ {unknown}
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
}                   //^ Trait::Item<T>
"#,
    );
}

#[test]
fn unselected_projection_in_trait_env_no_cycle() {
    // this is not a cycle
    check_types(
        r#"
//- /main.rs
trait Index {
    type Output;
}

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
}      //^ (UnificationStoreBase::Key<T>, UnificationStoreBase::Key<T>)
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
}       //^ u32
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
        "#,
        expect![[r#"
            1061..1072 '{ loop {} }': T
            1063..1070 'loop {}': !
            1068..1070 '{}': ()
            1136..1199 '{     ...     }': T
            1150..1155 'group': G
            1171..1175 'make': fn make<G>() -> G
            1171..1177 'make()': G
            1187..1191 'make': fn make<T>() -> T
            1187..1193 'make()': T
        "#]],
    );
}

#[test]
fn unify_impl_trait() {
    check_infer_with_mismatches(
        r#"
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
        "#,
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
            213..309 '{     ...t()) }': S<{unknown}>
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
            295..296 'S': S<{unknown}>({unknown}) -> S<{unknown}>
            295..307 'S(default())': S<{unknown}>
            297..304 'default': fn default<{unknown}>() -> {unknown}
            297..306 'default()': {unknown}
        "#]],
    );
}

#[test]
fn assoc_types_from_bounds() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            133..135 '_v': F
            178..181 '{ }': ()
            193..224 '{     ... }); }': ()
            199..209 'f::<(), _>': fn f<(), |&()| -> ()>(|&()| -> ())
            199..221 'f::<()... z; })': ()
            210..220 '|z| { z; }': |&()| -> ()
            211..212 'z': &()
            214..220 '{ z; }': ()
            216..217 'z': &()
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
}       //^ u32
"#,
    );
}

#[test]
fn dyn_trait_through_chalk() {
    check_types(
        r#"
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
    x.foo();
}       //^ ()
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
}               //^ String
"#,
    );
}

#[test]
fn iterator_chain() {
    check_infer_with_mismatches(
        r#"
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
        "#,
        expect![[r#"
            226..230 'self': Self
            232..233 'f': F
            317..328 '{ loop {} }': FilterMap<Self, F>
            319..326 'loop {}': !
            324..326 '{}': ()
            349..353 'self': Self
            355..356 'f': F
            405..416 '{ loop {} }': ()
            407..414 'loop {}': !
            412..414 '{}': ()
            525..529 'self': Self
            854..858 'self': I
            865..885 '{     ...     }': I
            875..879 'self': I
            944..955 '{ loop {} }': Vec<T>
            946..953 'loop {}': !
            951..953 '{}': ()
            1142..1269 '{     ... }); }': ()
            1148..1163 'Vec::<i32>::new': fn new<i32>() -> Vec<i32>
            1148..1165 'Vec::<...:new()': Vec<i32>
            1148..1177 'Vec::<...iter()': IntoIter<i32>
            1148..1240 'Vec::<...one })': FilterMap<IntoIter<i32>, |i32| -> Option<u32>>
            1148..1266 'Vec::<... y; })': ()
            1194..1239 '|x| if...None }': |i32| -> Option<u32>
            1195..1196 'x': i32
            1198..1239 'if x >...None }': Option<u32>
            1201..1202 'x': i32
            1201..1206 'x > 0': bool
            1205..1206 '0': i32
            1207..1225 '{ Some...u32) }': Option<u32>
            1209..1213 'Some': Some<u32>(u32) -> Option<u32>
            1209..1223 'Some(x as u32)': Option<u32>
            1214..1215 'x': i32
            1214..1222 'x as u32': u32
            1231..1239 '{ None }': Option<u32>
            1233..1237 'None': Option<u32>
            1255..1265 '|y| { y; }': |u32| -> ()
            1256..1257 'y': u32
            1259..1265 '{ y; }': ()
            1261..1262 'y': u32
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
}          //^ Foo
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
        }
        "#,
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
        "#,
        expect![[r#"
            110..114 'self': &Self
            166..267 '{     ...t(); }': ()
            172..178 'IsCopy': IsCopy
            172..185 'IsCopy.test()': bool
            191..198 'NotCopy': NotCopy
            191..205 'NotCopy.test()': {unknown}
            211..227 '(IsCop...sCopy)': (IsCopy, IsCopy)
            211..234 '(IsCop...test()': bool
            212..218 'IsCopy': IsCopy
            220..226 'IsCopy': IsCopy
            240..257 '(IsCop...tCopy)': (IsCopy, NotCopy)
            240..264 '(IsCop...test()': {unknown}
            241..247 'IsCopy': IsCopy
            249..256 'NotCopy': NotCopy
        "#]],
    );
}

#[test]
fn builtin_fn_def_copy() {
    check_infer_with_mismatches(
        r#"
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
        "#,
        expect![[r#"
            41..43 '{}': ()
            60..61 'T': {unknown}
            68..70 '{}': ()
            68..70: expected T, got ()
            145..149 'self': &Self
            201..281 '{     ...t(); }': ()
            207..210 'foo': fn foo()
            207..217 'foo.test()': bool
            223..226 'bar': fn bar<{unknown}>({unknown}) -> {unknown}
            223..233 'bar.test()': bool
            239..245 'Struct': Struct(usize) -> Struct
            239..252 'Struct.test()': bool
            258..271 'Enum::Variant': Variant(usize) -> Enum
            258..278 'Enum::...test()': bool
        "#]],
    );
}

#[test]
fn builtin_fn_ptr_copy() {
    check_infer_with_mismatches(
        r#"
        #[lang = "copy"]
        trait Copy {}

        trait Test { fn test(&self) -> bool; }
        impl<T: Copy> Test for T {}

        fn test(f1: fn(), f2: fn(usize) -> u8, f3: fn(u8, u8) -> &u8) {
            f1.test();
            f2.test();
            f3.test();
        }
        "#,
        expect![[r#"
            54..58 'self': &Self
            108..110 'f1': fn()
            118..120 'f2': fn(usize) -> u8
            139..141 'f3': fn(u8, u8) -> &u8
            162..210 '{     ...t(); }': ()
            168..170 'f1': fn()
            168..177 'f1.test()': bool
            183..185 'f2': fn(usize) -> u8
            183..192 'f2.test()': bool
            198..200 'f3': fn(u8, u8) -> &u8
            198..207 'f3.test()': bool
        "#]],
    );
}

#[test]
fn builtin_sized() {
    check_infer_with_mismatches(
        r#"
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
        "#,
        expect![[r#"
            56..60 'self': &Self
            113..228 '{     ...ized }': ()
            119..122 '1u8': u8
            119..129 '1u8.test()': bool
            135..150 '(*"foo").test()': {unknown}
            136..142 '*"foo"': str
            137..142 '"foo"': &str
            169..179 '(1u8, 1u8)': (u8, u8)
            169..186 '(1u8, ...test()': bool
            170..173 '1u8': u8
            175..178 '1u8': u8
            192..205 '(1u8, *"foo")': (u8, str)
            192..212 '(1u8, ...test()': {unknown}
            193..196 '1u8': u8
            198..204 '*"foo"': str
            199..204 '"foo"': &str
        "#]],
    );
}

#[test]
fn integer_range_iterate() {
    check_types(
        r#"
//- /main.rs crate:main deps:core
fn test() {
    for x in 0..100 { x; }
}                   //^ i32

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
        }
        "#,
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
fn infer_fn_trait_arg() {
    check_infer_with_mismatches(
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
        "#,
        expect![[r#"
            101..105 'self': &Self
            107..111 'args': Args
            220..224 'self': &Self
            226..230 'args': Args
            313..314 'f': F
            359..389 '{     ...f(s) }': T
            369..370 's': Option<i32>
            373..377 'None': Option<i32>
            383..384 'f': F
            383..387 'f(s)': T
            385..386 's': Option<i32>
        "#]],
    );
}

#[test]
fn infer_box_fn_arg() {
    // The type mismatch is a bug
    check_infer_with_mismatches(
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
            f(&s);
        }
        "#,
        expect![[r#"
            100..104 'self': Self
            106..110 'args': Args
            214..218 'self': &Self
            384..388 'self': &Box<T>
            396..423 '{     ...     }': &T
            406..417 '&self.inner': &*mut T
            407..411 'self': &Box<T>
            407..417 'self.inner': *mut T
            478..576 '{     ...&s); }': ()
            488..489 's': Option<i32>
            492..504 'Option::None': Option<i32>
            514..515 'f': Box<dyn FnOnce(&Option<i32>)>
            549..562 'box (|ps| {})': Box<|{unknown}| -> ()>
            554..561 '|ps| {}': |{unknown}| -> ()
            555..557 'ps': {unknown}
            559..561 '{}': ()
            568..569 'f': Box<dyn FnOnce(&Option<i32>)>
            568..573 'f(&s)': FnOnce::Output<dyn FnOnce(&Option<i32>), (&Option<i32>,)>
            570..572 '&s': &Option<i32>
            571..572 's': Option<i32>
            549..562: expected Box<dyn FnOnce(&Option<i32>)>, got Box<|_| -> ()>
        "#]],
    );
}

#[test]
fn infer_dyn_fn_output() {
    check_types(
        r#"
#[lang = "fn_once"]
pub trait FnOnce<Args> {
    type Output;
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

#[lang = "fn"]
pub trait Fn<Args>: FnOnce<Args> {
    extern "rust-call" fn call(&self, args: Args) -> Self::Output;
}

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
#[lang = "fn_once"]
pub trait FnOnce<Args> {
    type Output;
    extern "rust-call" fn call_once(self, args: Args) -> Self::Output;
}

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
  //^^^^^^^^ f32
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
}
        "#,
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
}
        "#,
        expect![[r#"
            17..73 '{     ...   } }': ()
            39..71 '{     ...     }': ()
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
}
        "#,
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
}

        "#,
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
}
    "#,
    );
}

#[test]
fn bin_op_adt_with_rhs_primitive() {
    check_infer_with_mismatches(
        r#"
#[lang = "add"]
pub trait Add<Rhs = Self> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}

struct Wrapper(u32);
impl Add<u32> for Wrapper {
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
            72..76 'self': Self
            78..81 'rhs': Rhs
            192..196 'self': Wrapper
            198..201 'rhs': u32
            219..247 '{     ...     }': Wrapper
            229..236 'Wrapper': Wrapper(u32) -> Wrapper
            229..241 'Wrapper(rhs)': Wrapper
            237..240 'rhs': u32
            259..345 '{     ...um;  }': ()
            269..276 'wrapped': Wrapper
            279..286 'Wrapper': Wrapper(u32) -> Wrapper
            279..290 'Wrapper(10)': Wrapper
            287..289 '10': u32
            300..303 'num': u32
            311..312 '2': u32
            322..325 'res': Wrapper
            328..335 'wrapped': Wrapper
            328..341 'wrapped + num': Wrapper
            338..341 'num': u32
        "#]],
    )
}
