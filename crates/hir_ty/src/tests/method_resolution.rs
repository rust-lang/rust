use expect_test::expect;

use super::{check_infer, check_types};

#[test]
fn infer_slice_method() {
    check_infer(
        r#"
        #[lang = "slice"]
        impl<T> [T] {
            fn foo(&self) -> T {
                loop {}
            }
        }

        #[lang = "slice_alloc"]
        impl<T> [T] {}

        fn test(x: &[u8]) {
            <[_]>::foo(x);
        }
        "#,
        expect![[r#"
            44..48 'self': &[T]
            55..78 '{     ...     }': T
            65..72 'loop {}': !
            70..72 '{}': ()
            130..131 'x': &[u8]
            140..162 '{     ...(x); }': ()
            146..156 '<[_]>::foo': fn foo<u8>(&[u8]) -> u8
            146..159 '<[_]>::foo(x)': u8
            157..158 'x': &[u8]
        "#]],
    );
}

#[test]
fn infer_associated_method_struct() {
    check_infer(
        r#"
        struct A { x: u32 }

        impl A {
            fn new() -> A {
                A { x: 0 }
            }
        }
        fn test() {
            let a = A::new();
            a.x;
        }
        "#,
        expect![[r#"
            48..74 '{     ...     }': A
            58..68 'A { x: 0 }': A
            65..66 '0': u32
            87..121 '{     ...a.x; }': ()
            97..98 'a': A
            101..107 'A::new': fn new() -> A
            101..109 'A::new()': A
            115..116 'a': A
            115..118 'a.x': u32
        "#]],
    );
}

#[test]
fn infer_associated_method_enum() {
    check_infer(
        r#"
        enum A { B, C }

        impl A {
            pub fn b() -> A {
                A::B
            }
            pub fn c() -> A {
                A::C
            }
        }
        fn test() {
            let a = A::b();
            a;
            let c = A::c();
            c;
        }
        "#,
        expect![[r#"
            46..66 '{     ...     }': A
            56..60 'A::B': A
            87..107 '{     ...     }': A
            97..101 'A::C': A
            120..177 '{     ...  c; }': ()
            130..131 'a': A
            134..138 'A::b': fn b() -> A
            134..140 'A::b()': A
            146..147 'a': A
            157..158 'c': A
            161..165 'A::c': fn c() -> A
            161..167 'A::c()': A
            173..174 'c': A
        "#]],
    );
}

#[test]
fn infer_associated_method_with_modules() {
    check_infer(
        r#"
        mod a {
            struct A;
            impl A { pub fn thing() -> A { A {} }}
        }

        mod b {
            struct B;
            impl B { pub fn thing() -> u32 { 99 }}

            mod c {
                struct C;
                impl C { pub fn thing() -> C { C {} }}
            }
        }
        use b::c;

        fn test() {
            let x = a::A::thing();
            let y = b::B::thing();
            let z = c::C::thing();
        }
        "#,
        expect![[r#"
            55..63 '{ A {} }': A
            57..61 'A {}': A
            125..131 '{ 99 }': u32
            127..129 '99': u32
            201..209 '{ C {} }': C
            203..207 'C {}': C
            240..324 '{     ...g(); }': ()
            250..251 'x': A
            254..265 'a::A::thing': fn thing() -> A
            254..267 'a::A::thing()': A
            277..278 'y': u32
            281..292 'b::B::thing': fn thing() -> u32
            281..294 'b::B::thing()': u32
            304..305 'z': C
            308..319 'c::C::thing': fn thing() -> C
            308..321 'c::C::thing()': C
        "#]],
    );
}

#[test]
fn infer_associated_method_generics() {
    check_infer(
        r#"
        struct Gen<T> {
            val: T
        }

        impl<T> Gen<T> {
            pub fn make(val: T) -> Gen<T> {
                Gen { val }
            }
        }

        fn test() {
            let a = Gen::make(0u32);
        }
        "#,
        expect![[r#"
            63..66 'val': T
            81..108 '{     ...     }': Gen<T>
            91..102 'Gen { val }': Gen<T>
            97..100 'val': T
            122..154 '{     ...32); }': ()
            132..133 'a': Gen<u32>
            136..145 'Gen::make': fn make<u32>(u32) -> Gen<u32>
            136..151 'Gen::make(0u32)': Gen<u32>
            146..150 '0u32': u32
        "#]],
    );
}

#[test]
fn infer_associated_method_generics_without_args() {
    check_infer(
        r#"
        struct Gen<T> {
            val: T
        }

        impl<T> Gen<T> {
            pub fn make() -> Gen<T> {
                loop { }
            }
        }

        fn test() {
            let a = Gen::<u32>::make();
        }
        "#,
        expect![[r#"
            75..99 '{     ...     }': Gen<T>
            85..93 'loop { }': !
            90..93 '{ }': ()
            113..148 '{     ...e(); }': ()
            123..124 'a': Gen<u32>
            127..143 'Gen::<...::make': fn make<u32>() -> Gen<u32>
            127..145 'Gen::<...make()': Gen<u32>
        "#]],
    );
}

#[test]
fn infer_associated_method_generics_2_type_params_without_args() {
    check_infer(
        r#"
        struct Gen<T, U> {
            val: T,
            val2: U,
        }

        impl<T> Gen<u32, T> {
            pub fn make() -> Gen<u32,T> {
                loop { }
            }
        }

        fn test() {
            let a = Gen::<u32, u64>::make();
        }
        "#,
        expect![[r#"
            101..125 '{     ...     }': Gen<u32, T>
            111..119 'loop { }': !
            116..119 '{ }': ()
            139..179 '{     ...e(); }': ()
            149..150 'a': Gen<u32, u64>
            153..174 'Gen::<...::make': fn make<u64>() -> Gen<u32, u64>
            153..176 'Gen::<...make()': Gen<u32, u64>
        "#]],
    );
}

#[test]
fn cross_crate_associated_method_call() {
    check_types(
        r#"
//- /main.rs crate:main deps:other_crate
fn test() {
    let x = other_crate::foo::S::thing();
    x;
} //^ i128

//- /lib.rs crate:other_crate
mod foo {
    struct S;
    impl S {
        fn thing() -> i128 {}
    }
}
"#,
    );
}

#[test]
fn infer_trait_method_simple() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_infer(
        r#"
        trait Trait1 {
            fn method(&self) -> u32;
        }
        struct S1;
        impl Trait1 for S1 {}
        trait Trait2 {
            fn method(&self) -> i128;
        }
        struct S2;
        impl Trait2 for S2 {}
        fn test() {
            S1.method(); // -> u32
            S2.method(); // -> i128
        }
        "#,
        expect![[r#"
            30..34 'self': &Self
            109..113 'self': &Self
            169..227 '{     ...i128 }': ()
            175..177 'S1': S1
            175..186 'S1.method()': u32
            202..204 'S2': S2
            202..213 'S2.method()': i128
        "#]],
    );
}

#[test]
fn infer_trait_method_scoped() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_infer(
        r#"
        struct S;
        mod foo {
            pub trait Trait1 {
                fn method(&self) -> u32;
            }
            impl Trait1 for super::S {}
        }
        mod bar {
            pub trait Trait2 {
                fn method(&self) -> i128;
            }
            impl Trait2 for super::S {}
        }

        mod foo_test {
            use super::S;
            use super::foo::Trait1;
            fn test() {
                S.method(); // -> u32
            }
        }

        mod bar_test {
            use super::S;
            use super::bar::Trait2;
            fn test() {
                S.method(); // -> i128
            }
        }
        "#,
        expect![[r#"
            62..66 'self': &Self
            168..172 'self': &Self
            299..336 '{     ...     }': ()
            309..310 'S': S
            309..319 'S.method()': u32
            415..453 '{     ...     }': ()
            425..426 'S': S
            425..435 'S.method()': i128
        "#]],
    );
}

#[test]
fn infer_trait_method_generic_1() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_infer(
        r#"
        trait Trait<T> {
            fn method(&self) -> T;
        }
        struct S;
        impl Trait<u32> for S {}
        fn test() {
            S.method();
        }
        "#,
        expect![[r#"
            32..36 'self': &Self
            91..110 '{     ...d(); }': ()
            97..98 'S': S
            97..107 'S.method()': u32
        "#]],
    );
}

#[test]
fn infer_trait_method_generic_more_params() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_infer(
        r#"
        trait Trait<T1, T2, T3> {
            fn method1(&self) -> (T1, T2, T3);
            fn method2(&self) -> (T3, T2, T1);
        }
        struct S1;
        impl Trait<u8, u16, u32> for S1 {}
        struct S2;
        impl<T> Trait<i8, i16, T> for S2 {}
        fn test() {
            S1.method1(); // u8, u16, u32
            S1.method2(); // u32, u16, u8
            S2.method1(); // i8, i16, {unknown}
            S2.method2(); // {unknown}, i16, i8
        }
        "#,
        expect![[r#"
            42..46 'self': &Self
            81..85 'self': &Self
            209..360 '{     ..., i8 }': ()
            215..217 'S1': S1
            215..227 'S1.method1()': (u8, u16, u32)
            249..251 'S1': S1
            249..261 'S1.method2()': (u32, u16, u8)
            283..285 'S2': S2
            283..295 'S2.method1()': (i8, i16, {unknown})
            323..325 'S2': S2
            323..335 'S2.method2()': ({unknown}, i16, i8)
        "#]],
    );
}

#[test]
fn infer_trait_method_generic_2() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_infer(
        r#"
        trait Trait<T> {
            fn method(&self) -> T;
        }
        struct S<T>(T);
        impl<U> Trait<U> for S<U> {}
        fn test() {
            S(1u32).method();
        }
        "#,
        expect![[r#"
            32..36 'self': &Self
            101..126 '{     ...d(); }': ()
            107..108 'S': S<u32>(u32) -> S<u32>
            107..114 'S(1u32)': S<u32>
            107..123 'S(1u32...thod()': u32
            109..113 '1u32': u32
        "#]],
    );
}

#[test]
fn infer_trait_assoc_method() {
    check_infer(
        r#"
        trait Default {
            fn default() -> Self;
        }
        struct S;
        impl Default for S {}
        fn test() {
            let s1: S = Default::default();
            let s2 = S::default();
            let s3 = <S as Default>::default();
        }
        "#,
        expect![[r#"
            86..192 '{     ...t(); }': ()
            96..98 's1': S
            104..120 'Defaul...efault': fn default<S>() -> S
            104..122 'Defaul...ault()': S
            132..134 's2': S
            137..147 'S::default': fn default<S>() -> S
            137..149 'S::default()': S
            159..161 's3': S
            164..187 '<S as ...efault': fn default<S>() -> S
            164..189 '<S as ...ault()': S
        "#]],
    );
}

#[test]
fn infer_trait_assoc_method_generics_1() {
    check_infer(
        r#"
        trait Trait<T> {
            fn make() -> T;
        }
        struct S;
        impl Trait<u32> for S {}
        struct G<T>;
        impl<T> Trait<T> for G<T> {}
        fn test() {
            let a = S::make();
            let b = G::<u64>::make();
            let c: f64 = G::make();
        }
        "#,
        expect![[r#"
            126..210 '{     ...e(); }': ()
            136..137 'a': u32
            140..147 'S::make': fn make<S, u32>() -> u32
            140..149 'S::make()': u32
            159..160 'b': u64
            163..177 'G::<u64>::make': fn make<G<u64>, u64>() -> u64
            163..179 'G::<u6...make()': u64
            189..190 'c': f64
            198..205 'G::make': fn make<G<f64>, f64>() -> f64
            198..207 'G::make()': f64
        "#]],
    );
}

#[test]
fn infer_trait_assoc_method_generics_2() {
    check_infer(
        r#"
        trait Trait<T> {
            fn make<U>() -> (T, U);
        }
        struct S;
        impl Trait<u32> for S {}
        struct G<T>;
        impl<T> Trait<T> for G<T> {}
        fn test() {
            let a = S::make::<i64>();
            let b: (_, i64) = S::make();
            let c = G::<u32>::make::<i64>();
            let d: (u32, _) = G::make::<i64>();
            let e: (u32, i64) = G::make();
        }
        "#,
        expect![[r#"
            134..312 '{     ...e(); }': ()
            144..145 'a': (u32, i64)
            148..162 'S::make::<i64>': fn make<S, u32, i64>() -> (u32, i64)
            148..164 'S::mak...i64>()': (u32, i64)
            174..175 'b': (u32, i64)
            188..195 'S::make': fn make<S, u32, i64>() -> (u32, i64)
            188..197 'S::make()': (u32, i64)
            207..208 'c': (u32, i64)
            211..232 'G::<u3...:<i64>': fn make<G<u32>, u32, i64>() -> (u32, i64)
            211..234 'G::<u3...i64>()': (u32, i64)
            244..245 'd': (u32, i64)
            258..272 'G::make::<i64>': fn make<G<u32>, u32, i64>() -> (u32, i64)
            258..274 'G::mak...i64>()': (u32, i64)
            284..285 'e': (u32, i64)
            300..307 'G::make': fn make<G<u32>, u32, i64>() -> (u32, i64)
            300..309 'G::make()': (u32, i64)
        "#]],
    );
}

#[test]
fn infer_trait_assoc_method_generics_3() {
    check_infer(
        r#"
        trait Trait<T> {
            fn make() -> (Self, T);
        }
        struct S<T>;
        impl Trait<i64> for S<i32> {}
        fn test() {
            let a = S::make();
        }
        "#,
        expect![[r#"
            100..126 '{     ...e(); }': ()
            110..111 'a': (S<i32>, i64)
            114..121 'S::make': fn make<S<i32>, i64>() -> (S<i32>, i64)
            114..123 'S::make()': (S<i32>, i64)
        "#]],
    );
}

#[test]
fn infer_trait_assoc_method_generics_4() {
    check_infer(
        r#"
        trait Trait<T> {
            fn make() -> (Self, T);
        }
        struct S<T>;
        impl Trait<i64> for S<u64> {}
        impl Trait<i32> for S<u32> {}
        fn test() {
            let a: (S<u64>, _) = S::make();
            let b: (_, i32) = S::make();
        }
        "#,
        expect![[r#"
            130..202 '{     ...e(); }': ()
            140..141 'a': (S<u64>, i64)
            157..164 'S::make': fn make<S<u64>, i64>() -> (S<u64>, i64)
            157..166 'S::make()': (S<u64>, i64)
            176..177 'b': (S<u32>, i32)
            190..197 'S::make': fn make<S<u32>, i32>() -> (S<u32>, i32)
            190..199 'S::make()': (S<u32>, i32)
        "#]],
    );
}

#[test]
fn infer_trait_assoc_method_generics_5() {
    check_infer(
        r#"
        trait Trait<T> {
            fn make<U>() -> (Self, T, U);
        }
        struct S<T>;
        impl Trait<i64> for S<u64> {}
        fn test() {
            let a = <S as Trait<i64>>::make::<u8>();
            let b: (S<u64>, _, _) = Trait::<i64>::make::<u8>();
        }
        "#,
        expect![[r#"
            106..210 '{     ...>(); }': ()
            116..117 'a': (S<u64>, i64, u8)
            120..149 '<S as ...::<u8>': fn make<S<u64>, i64, u8>() -> (S<u64>, i64, u8)
            120..151 '<S as ...<u8>()': (S<u64>, i64, u8)
            161..162 'b': (S<u64>, i64, u8)
            181..205 'Trait:...::<u8>': fn make<S<u64>, i64, u8>() -> (S<u64>, i64, u8)
            181..207 'Trait:...<u8>()': (S<u64>, i64, u8)
        "#]],
    );
}

#[test]
fn infer_call_trait_method_on_generic_param_1() {
    check_infer(
        r#"
        trait Trait {
            fn method(&self) -> u32;
        }
        fn test<T: Trait>(t: T) {
            t.method();
        }
        "#,
        expect![[r#"
            29..33 'self': &Self
            63..64 't': T
            69..88 '{     ...d(); }': ()
            75..76 't': T
            75..85 't.method()': u32
        "#]],
    );
}

#[test]
fn infer_call_trait_method_on_generic_param_2() {
    check_infer(
        r#"
        trait Trait<T> {
            fn method(&self) -> T;
        }
        fn test<U, T: Trait<U>>(t: T) {
            t.method();
        }
        "#,
        expect![[r#"
            32..36 'self': &Self
            70..71 't': T
            76..95 '{     ...d(); }': ()
            82..83 't': T
            82..92 't.method()': U
        "#]],
    );
}

#[test]
fn infer_with_multiple_trait_impls() {
    check_infer(
        r#"
        trait Into<T> {
            fn into(self) -> T;
        }
        struct S;
        impl Into<u32> for S {}
        impl Into<u64> for S {}
        fn test() {
            let x: u32 = S.into();
            let y: u64 = S.into();
            let z = Into::<u64>::into(S);
        }
        "#,
        expect![[r#"
            28..32 'self': Self
            110..201 '{     ...(S); }': ()
            120..121 'x': u32
            129..130 'S': S
            129..137 'S.into()': u32
            147..148 'y': u64
            156..157 'S': S
            156..164 'S.into()': u64
            174..175 'z': u64
            178..195 'Into::...::into': fn into<S, u64>(S) -> u64
            178..198 'Into::...nto(S)': u64
            196..197 'S': S
        "#]],
    );
}

#[test]
fn method_resolution_unify_impl_self_type() {
    check_types(
        r#"
struct S<T>;
impl S<u32> { fn foo(&self) -> u8 {} }
impl S<i32> { fn foo(&self) -> i8 {} }
fn test() { (S::<u32>.foo(), S::<i32>.foo()); }
          //^ (u8, i8)
"#,
    );
}

#[test]
fn method_resolution_trait_before_autoref() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(&self) -> i8 { 0 } }
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { S.foo(); }
                //^ u128
"#,
    );
}

#[test]
fn method_resolution_by_value_before_autoref() {
    check_types(
        r#"
trait Clone { fn clone(&self) -> Self; }
struct S;
impl Clone for S {}
impl Clone for &S {}
fn test() { (S.clone(), (&S).clone(), (&&S).clone()); }
          //^ (S, S, &S)
"#,
    );
}

#[test]
fn method_resolution_trait_before_autoderef() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(self) -> i8 { 0 } }
impl Trait for &S { fn foo(self) -> u128 { 0 } }
fn test() { (&S).foo(); }
                   //^ u128
"#,
    );
}

#[test]
fn method_resolution_impl_before_trait() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(self) -> i8 { 0 } }
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { S.foo(); }
                //^ i8
"#,
    );
}

#[test]
fn method_resolution_impl_ref_before_trait() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(&self) -> i8 { 0 } }
impl Trait for &S { fn foo(self) -> u128 { 0 } }
fn test() { S.foo(); }
                //^ i8
"#,
    );
}

#[test]
fn method_resolution_trait_autoderef() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { (&S).foo(); }
                   //^ u128
"#,
    );
}

#[test]
fn method_resolution_unsize_array() {
    check_types(
        r#"
#[lang = "slice"]
impl<T> [T] {
    fn len(&self) -> usize { loop {} }
}
fn test() {
    let a = [1, 2, 3];
    a.len();
}       //^ usize
"#,
    );
}

#[test]
fn method_resolution_trait_from_prelude() {
    check_types(
        r#"
//- /main.rs crate:main deps:other_crate
struct S;
impl Clone for S {}

fn test() {
    S.clone();
          //^ S
}

//- /lib.rs crate:other_crate
#[prelude_import] use foo::*;

mod foo {
    trait Clone {
        fn clone(&self) -> Self;
    }
}
"#,
    );
}

#[test]
fn method_resolution_where_clause_for_unknown_trait() {
    // The blanket impl currently applies because we ignore the unresolved where clause
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T> Trait for T where T: UnknownTrait {}
fn test() { (&S).foo(); }
                   //^ u128
"#,
    );
}

#[test]
fn method_resolution_where_clause_not_met() {
    // The blanket impl shouldn't apply because we can't prove S: Clone
    // This is also to make sure that we don't resolve to the foo method just
    // because that's the only method named foo we can find, which would make
    // the below tests not work
    check_types(
        r#"
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T> Trait for T where T: Clone {}
fn test() { (&S).foo(); }
                   //^ {unknown}
"#,
    );
}

#[test]
fn method_resolution_where_clause_inline_not_met() {
    // The blanket impl shouldn't apply because we can't prove S: Clone
    check_types(
        r#"
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T: Clone> Trait for T {}
fn test() { (&S).foo(); }
                   //^ {unknown}
"#,
    );
}

#[test]
fn method_resolution_where_clause_1() {
    check_types(
        r#"
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl Clone for S {}
impl<T> Trait for T where T: Clone {}
fn test() { S.foo(); }
                //^ u128
"#,
    );
}

#[test]
fn method_resolution_where_clause_2() {
    check_types(
        r#"
trait Into<T> { fn into(self) -> T; }
trait From<T> { fn from(other: T) -> Self; }
struct S1;
struct S2;
impl From<S2> for S1 {}
impl<T, U> Into<U> for T where U: From<T> {}
fn test() { S2.into(); }
                  //^ {unknown}
"#,
    );
}

#[test]
fn method_resolution_where_clause_inline() {
    check_types(
        r#"
trait Into<T> { fn into(self) -> T; }
trait From<T> { fn from(other: T) -> Self; }
struct S1;
struct S2;
impl From<S2> for S1 {}
impl<T, U: From<T>> Into<U> for T {}
fn test() { S2.into(); }
                  //^ {unknown}
"#,
    );
}

#[test]
fn method_resolution_overloaded_method() {
    cov_mark::check!(impl_self_type_match_without_receiver);
    check_types(
        r#"
struct Wrapper<T>(T);
struct Foo<T>(T);
struct Bar<T>(T);

impl<T> Wrapper<Foo<T>> {
    pub fn new(foo_: T) -> Self {
        Wrapper(Foo(foo_))
    }
}

impl<T> Wrapper<Bar<T>> {
    pub fn new(bar_: T) -> Self {
        Wrapper(Bar(bar_))
    }
}

fn main() {
    let a = Wrapper::<Foo<f32>>::new(1.0);
    let b = Wrapper::<Bar<f32>>::new(1.0);
    (a, b);
  //^ (Wrapper<Foo<f32>>, Wrapper<Bar<f32>>)
}
"#,
    );
}

#[test]
fn method_resolution_encountering_fn_type() {
    check_types(
        r#"
//- /main.rs
fn foo() {}
trait FnOnce { fn call(self); }
fn test() { foo.call(); }
                   //^ {unknown}
"#,
    );
}

#[test]
fn super_trait_impl_return_trait_method_resolution() {
    check_infer(
        r#"
        trait Base {
            fn foo(self) -> usize;
        }

        trait Super : Base {}

        fn base1() -> impl Base { loop {} }
        fn super1() -> impl Super { loop {} }

        fn test(base2: impl Base, super2: impl Super) {
            base1().foo();
            super1().foo();
            base2.foo();
            super2.foo();
        }
        "#,
        expect![[r#"
            24..28 'self': Self
            90..101 '{ loop {} }': !
            92..99 'loop {}': !
            97..99 '{}': ()
            128..139 '{ loop {} }': !
            130..137 'loop {}': !
            135..137 '{}': ()
            149..154 'base2': impl Base
            167..173 'super2': impl Super
            187..264 '{     ...o(); }': ()
            193..198 'base1': fn base1() -> impl Base
            193..200 'base1()': impl Base
            193..206 'base1().foo()': usize
            212..218 'super1': fn super1() -> impl Super
            212..220 'super1()': impl Super
            212..226 'super1().foo()': usize
            232..237 'base2': impl Base
            232..243 'base2.foo()': usize
            249..255 'super2': impl Super
            249..261 'super2.foo()': usize
        "#]],
    );
}

#[test]
fn method_resolution_non_parameter_type() {
    check_types(
        r#"
mod a {
    pub trait Foo {
        fn foo(&self);
    }
}

struct Wrapper<T>(T);
fn foo<T>(t: Wrapper<T>)
where
    Wrapper<T>: a::Foo,
{
    t.foo();
}       //^ {unknown}
"#,
    );
}

#[test]
fn method_resolution_3373() {
    check_types(
        r#"
struct A<T>(T);

impl A<i32> {
    fn from(v: i32) -> A<i32> { A(v) }
}

fn main() {
    A::from(3);
}          //^ A<i32>
"#,
    );
}

#[test]
fn method_resolution_slow() {
    // this can get quite slow if we set the solver size limit too high
    check_types(
        r#"
trait SendX {}

struct S1; impl SendX for S1 {}
struct S2; impl SendX for S2 {}
struct U1;

trait Trait { fn method(self); }

struct X1<A, B> {}
impl<A, B> SendX for X1<A, B> where A: SendX, B: SendX {}

struct S<B, C> {}

trait FnX {}

impl<B, C> Trait for S<B, C> where C: FnX, B: SendX {}

fn test() { (S {}).method(); }
                        //^ ()
"#,
    );
}

#[test]
fn dyn_trait_super_trait_not_in_scope() {
    check_infer(
        r#"
        mod m {
            pub trait SuperTrait {
                fn foo(&self) -> u32 { 0 }
            }
        }
        trait Trait: m::SuperTrait {}

        struct S;
        impl m::SuperTrait for S {}
        impl Trait for S {}

        fn test(d: &dyn Trait) {
            d.foo();
        }
        "#,
        expect![[r#"
            51..55 'self': &Self
            64..69 '{ 0 }': u32
            66..67 '0': u32
            176..177 'd': &dyn Trait
            191..207 '{     ...o(); }': ()
            197..198 'd': &dyn Trait
            197..204 'd.foo()': u32
        "#]],
    );
}

#[test]
fn method_resolution_foreign_opaque_type() {
    check_infer(
        r#"
        extern "C" {
            type S;
            fn f() -> &'static S;
        }

        impl S {
            fn foo(&self) -> bool {
                true
            }
        }

        fn test() {
            let s = unsafe { f() };
            s.foo();
        }
        "#,
        expect![[r#"
            75..79 'self': &S
            89..109 '{     ...     }': bool
            99..103 'true': bool
            123..167 '{     ...o(); }': ()
            133..134 's': &S
            137..151 'unsafe { f() }': &S
            144..151 '{ f() }': &S
            146..147 'f': fn f() -> &S
            146..149 'f()': &S
            157..158 's': &S
            157..164 's.foo()': bool
        "#]],
    );
}

#[test]
fn method_with_allocator_box_self_type() {
    check_types(
        r#"
struct Slice<T> {}
struct Box<T, A> {}

impl<T> Slice<T> {
    pub fn into_vec<A>(self: Box<Self, A>) { }
}

fn main() {
    let foo: Slice<u32>;
    (foo.into_vec()); // we don't actually support arbitrary self types, but we shouldn't crash at least
} //^ {unknown}
"#,
    );
}

#[test]
fn method_on_dyn_impl() {
    check_types(
        r#"
trait Foo {}

impl Foo for u32 {}
impl dyn Foo + '_ {
    pub fn dyn_foo(&self) -> u32 {
        0
    }
}

fn main() {
    let f = &42u32 as &dyn Foo;
    f.dyn_foo();
  // ^u32
}
"#,
    );
}

#[test]
fn autoderef_visibility_field() {
    check_infer(
        r#"
#[lang = "deref"]
pub trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}
mod a {
    pub struct Foo(pub char);
    pub struct Bar(i32);
    impl Bar {
        pub fn new() -> Self {
            Self(0)
        }
    }
    impl super::Deref for Bar {
        type Target = Foo;
        fn deref(&self) -> &Foo {
            &Foo('z')
        }
    }
}
mod b {
    fn foo() {
        let x = super::a::Bar::new().0;
    }
}
        "#,
        expect![[r#"
            67..71 'self': &Self
            200..231 '{     ...     }': Bar
            214..218 'Self': Bar(i32) -> Bar
            214..221 'Self(0)': Bar
            219..220 '0': i32
            315..319 'self': &Bar
            329..362 '{     ...     }': &Foo
            343..352 '&Foo('z')': &Foo
            344..347 'Foo': Foo(char) -> Foo
            344..352 'Foo('z')': Foo
            348..351 ''z'': char
            392..439 '{     ...     }': ()
            406..407 'x': char
            410..428 'super:...r::new': fn new() -> Bar
            410..430 'super:...:new()': Bar
            410..432 'super:...ew().0': char
        "#]],
    )
}

#[test]
fn autoderef_visibility_method() {
    cov_mark::check!(autoderef_candidate_not_visible);
    check_infer(
        r#"
#[lang = "deref"]
pub trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}
mod a {
    pub struct Foo(pub char);
    impl Foo {
        pub fn mango(&self) -> char {
            self.0
        }
    }
    pub struct Bar(i32);
    impl Bar {
        pub fn new() -> Self {
            Self(0)
        }
        fn mango(&self) -> i32 {
            self.0
        }
    }
    impl super::Deref for Bar {
        type Target = Foo;
        fn deref(&self) -> &Foo {
            &Foo('z')
        }
    }
}
mod b {
    fn foo() {
        let x = super::a::Bar::new().mango();
    }
}
        "#,
        expect![[r#"
            67..71 'self': &Self
            168..172 'self': &Foo
            182..212 '{     ...     }': char
            196..200 'self': &Foo
            196..202 'self.0': char
            288..319 '{     ...     }': Bar
            302..306 'Self': Bar(i32) -> Bar
            302..309 'Self(0)': Bar
            307..308 '0': i32
            338..342 'self': &Bar
            351..381 '{     ...     }': i32
            365..369 'self': &Bar
            365..371 'self.0': i32
            465..469 'self': &Bar
            479..512 '{     ...     }': &Foo
            493..502 '&Foo('z')': &Foo
            494..497 'Foo': Foo(char) -> Foo
            494..502 'Foo('z')': Foo
            498..501 ''z'': char
            542..595 '{     ...     }': ()
            556..557 'x': char
            560..578 'super:...r::new': fn new() -> Bar
            560..580 'super:...:new()': Bar
            560..588 'super:...ango()': char
        "#]],
    )
}

#[test]
fn trait_impl_in_unnamed_const() {
    check_types(
        r#"
struct S;

trait Tr {
    fn method(&self) -> u16;
}

const _: () = {
    impl Tr for S {}
};

fn f() {
    S.method();
  //^^^^^^^^^^ u16
}
    "#,
    );
}

#[test]
fn inherent_impl_in_unnamed_const() {
    check_types(
        r#"
struct S;

const _: () = {
    impl S {
        fn method(&self) -> u16 { 0 }

        pub(super) fn super_method(&self) -> u16 { 0 }

        pub(crate) fn crate_method(&self) -> u16 { 0 }

        pub fn pub_method(&self) -> u16 { 0 }
    }
};

fn f() {
    S.method();
  //^^^^^^^^^^ u16

    S.super_method();
  //^^^^^^^^^^^^^^^^ u16

    S.crate_method();
  //^^^^^^^^^^^^^^^^ u16

    S.pub_method();
  //^^^^^^^^^^^^^^ u16
}
    "#,
    );
}
