use expect_test::expect;

use crate::tests::check;

use super::{check_infer, check_no_mismatches, check_types};

#[test]
fn infer_slice_method() {
    check_types(
        r#"
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    fn foo(&self) -> T {
        loop {}
    }
}

fn test(x: &[u8]) {
    <[_]>::foo(x);
  //^^^^^^^^^^^^^ u8
}
        "#,
    );
}

#[test]
fn cross_crate_primitive_method() {
    check_types(
        r#"
//- /main.rs crate:main deps:other_crate
fn test() {
    let x = 1f32;
    x.foo();
} //^^^^^^^ f32

//- /lib.rs crate:other_crate
mod foo {
    impl f32 {
        #[rustc_allow_incoherent_impl]
        pub fn foo(self) -> f32 { 0. }
    }
}
"#,
    );
}

#[test]
fn infer_array_inherent_impl() {
    check_types(
        r#"
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    fn foo(&self) -> T {
        loop {}
    }
}
fn test(x: &[u8; 0]) {
    <[_; 0]>::foo(x);
  //^^^^^^^^^^^^^^^^ u8
}
        "#,
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
fn infer_associated_method_struct_in_local_scope() {
    check_infer(
        r#"
        fn mismatch() {
            struct A;

            impl A {
                fn from(_: i32, _: i32) -> Self {
                    A
                }
            }

            let _a = A::from(1, 2);
        }
        "#,
        expect![[r#"
            14..146 '{     ... 2); }': ()
            125..127 '_a': A
            130..137 'A::from': fn from(i32, i32) -> A
            130..143 'A::from(1, 2)': A
            138..139 '1': i32
            141..142 '2': i32
            60..61 '_': i32
            68..69 '_': i32
            84..109 '{     ...     }': A
            98..99 'A': A
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
            pub struct A;
            impl A { pub fn thing() -> A { A {} }}
        }

        mod b {
            pub struct B;
            impl B { pub fn thing() -> u32 { 99 }}

            pub mod c {
                pub struct C;
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
            59..67 '{ A {} }': A
            61..65 'A {}': A
            133..139 '{ 99 }': u32
            135..137 '99': u32
            217..225 '{ C {} }': C
            219..223 'C {}': C
            256..340 '{     ...g(); }': ()
            266..267 'x': A
            270..281 'a::A::thing': fn thing() -> A
            270..283 'a::A::thing()': A
            293..294 'y': u32
            297..308 'b::B::thing': fn thing() -> u32
            297..310 'b::B::thing()': u32
            320..321 'z': C
            324..335 'c::C::thing': fn thing() -> C
            324..337 'c::C::thing()': C
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
pub mod foo {
    pub struct S;
    impl S {
        pub fn thing() -> i128 { 0 }
    }
}
"#,
    );
}

#[test]
fn infer_trait_method_simple() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_types(
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
    S1.method();
  //^^^^^^^^^^^ u32
    S2.method(); // -> i128
  //^^^^^^^^^^^ i128
}
        "#,
    );
}

#[test]
fn infer_trait_method_scoped() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_types(
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
        S.method();
      //^^^^^^^^^^ u32
    }
}

mod bar_test {
    use super::S;
    use super::bar::Trait2;
    fn test() {
        S.method();
      //^^^^^^^^^^ i128
    }
}
        "#,
    );
}

#[test]
fn infer_trait_method_multiple_mutable_reference() {
    check_types(
        r#"
trait Trait {
    fn method(&mut self) -> i32 { 5 }
}
struct S;
impl Trait for &mut &mut S {}
fn test() {
    let s = &mut &mut &mut S;
    s.method();
  //^^^^^^^^^^ i32
}
        "#,
    );
}

#[test]
fn infer_trait_method_generic_1() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_types(
        r#"
trait Trait<T> {
    fn method(&self) -> T;
}
struct S;
impl Trait<u32> for S {}
fn test() {
    S.method();
  //^^^^^^^^^^ u32
}
        "#,
    );
}

#[test]
fn infer_trait_method_generic_more_params() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_types(
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
    S1.method1();
  //^^^^^^^^^^^^ (u8, u16, u32)
    S1.method2();
  //^^^^^^^^^^^^ (u32, u16, u8)
    S2.method1();
  //^^^^^^^^^^^^ (i8, i16, {unknown})
    S2.method2();
  //^^^^^^^^^^^^ ({unknown}, i16, i8)
}
        "#,
    );
}

#[test]
fn infer_trait_method_generic_2() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    check_types(
        r#"
trait Trait<T> {
    fn method(&self) -> T;
}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn test() {
    S(1u32).method();
  //^^^^^^^^^^^^^^^^ u32
}
        "#,
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
impl S<u32> { fn foo(&self) -> u8 { 0 } }
impl S<i32> { fn foo(&self) -> i8 { 0 } }
fn test() { (S::<u32>.foo(), S::<i32>.foo()); }
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ (u8, i8)
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
          //^^^^^^^ u128
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
          //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ (S, S, &S)
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
          //^^^^^^^^^^ u128
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
          //^^^^^^^ i8
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
          //^^^^^^^ i8
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
          //^^^^^^^^^^ u128
"#,
    );
}

#[test]
fn method_resolution_unsize_array() {
    check_types(
        r#"
//- minicore: slice
fn test() {
    let a = [1, 2, 3];
    a.len();
} //^^^^^^^ usize
"#,
    );
}

#[test]
fn method_resolution_trait_from_prelude() {
    check_types(
        r#"
//- /main.rs edition:2018 crate:main deps:core
struct S;
impl Clone for S {}

fn test() {
    S.clone();
  //^^^^^^^^^ S
}

//- /lib.rs crate:core
pub mod prelude {
    pub mod rust_2018 {
        pub trait Clone {
            fn clone(&self) -> Self;
        }
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
          //^^^^^^^^^^ u128
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
          //^^^^^^^^^^ {unknown}
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
          //^^^^^^^^^^ {unknown}
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
          //^^^^^^^ u128
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
          //^^^^^^^^^ {unknown}
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
          //^^^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn method_resolution_overloaded_method() {
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
  //^^^^^^ (Wrapper<Foo<f32>>, Wrapper<Bar<f32>>)
}
"#,
    );
}

#[test]
fn method_resolution_overloaded_const() {
    cov_mark::check!(const_candidate_self_type_mismatch);
    check_types(
        r#"
struct Wrapper<T>(T);
struct Foo<T>(T);
struct Bar<T>(T);

impl<T> Wrapper<Foo<T>> {
    pub const VALUE: Foo<T>;
}

impl<T> Wrapper<Bar<T>> {
    pub const VALUE: Bar<T>;
}

fn main() {
    let a = Wrapper::<Foo<f32>>::VALUE;
    let b = Wrapper::<Bar<f32>>::VALUE;
    (a, b);
  //^^^^^^ (Foo<f32>, Bar<f32>)
}
"#,
    );
}

#[test]
fn explicit_fn_once_call_fn_item() {
    check_types(
        r#"
//- minicore: fn
fn foo() {}
fn test() { foo.call_once(); }
          //^^^^^^^^^^^^^^^ ()
"#,
    );
}

#[test]
fn super_trait_impl_return_trait_method_resolution() {
    check_infer(
        r#"
        //- minicore: sized
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
} //^^^^^^^ {unknown}
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
} //^^^^^^^^^^ A<i32>
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
          //^^^^^^^^^^^^^^^ ()
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
    foo.into_vec(); // we shouldn't crash on this at least
} //^^^^^^^^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn inherent_method_deref_raw() {
    check_types(
        r#"
struct Val;

impl Val {
    pub fn method(self: *const Val) -> u32 {
        0
    }
}

fn main() {
    let foo: *const Val;
    foo.method();
 // ^^^^^^^^^^^^ u32
}
"#,
    );
}

#[test]
fn inherent_method_ref_self_deref_raw() {
    check_types(
        r#"
struct Val;

impl Val {
    pub fn method(&self) -> u32 {
        0
    }
}

fn main() {
    let foo: *const Val;
    foo.method();
 // ^^^^^^^^^^^^ {unknown}
}
"#,
    );
}

#[test]
fn trait_method_deref_raw() {
    check_types(
        r#"
trait Trait {
    fn method(self: *const Self) -> u32;
}

struct Val;

impl Trait for Val {
    fn method(self: *const Self) -> u32 {
        0
    }
}

fn main() {
    let foo: *const Val;
    foo.method();
 // ^^^^^^^^^^^^ u32
}
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
 // ^^^^^^^^^^^ u32
}
"#,
    );
}

#[test]
fn dyn_trait_method_priority() {
    check_types(
        r#"
//- minicore: from
trait Trait {
    fn into(&self) -> usize { 0 }
}

fn foo(a: &dyn Trait) {
    let _ = a.into();
      //^usize
}
        "#,
    );
}

#[test]
fn trait_method_priority_for_placeholder_type() {
    check_types(
        r#"
//- minicore: from
trait Trait {
    fn into(&self) -> usize { 0 }
}

fn foo<T: Trait>(a: &T) {
    let _ = a.into();
      //^usize
}
        "#,
    );
}

#[test]
fn autoderef_visibility_field() {
    check(
        r#"
//- minicore: deref
mod a {
    pub struct Foo(pub char);
    pub struct Bar(i32);
    impl Bar {
        pub fn new() -> Self {
            Self(0)
        }
    }
    impl core::ops::Deref for Bar {
        type Target = Foo;
        fn deref(&self) -> &Foo {
            &Foo('z')
        }
    }
}
mod b {
    fn foo() {
        let x = super::a::Bar::new().0;
             // ^^^^^^^^^^^^^^^^^^^^ adjustments: Deref(Some(OverloadedDeref(Some(Not))))
             // ^^^^^^^^^^^^^^^^^^^^^^ type: char
    }
}
"#,
    )
}

#[test]
fn autoderef_visibility_method() {
    cov_mark::check!(autoderef_candidate_not_visible);
    check(
        r#"
//- minicore: deref
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
    impl core::ops::Deref for Bar {
        type Target = Foo;
        fn deref(&self) -> &Foo {
            &Foo('z')
        }
    }
}
mod b {
    fn foo() {
        let x = super::a::Bar::new().mango();
             // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ type: char
    }
}
"#,
    )
}

#[test]
fn trait_vs_private_inherent_const() {
    cov_mark::check!(const_candidate_not_visible);
    check(
        r#"
mod a {
    pub struct Foo;
    impl Foo {
        const VALUE: u32 = 2;
    }
    pub trait Trait {
        const VALUE: usize;
    }
    impl Trait for Foo {
        const VALUE: usize = 3;
    }

    fn foo() {
        let x = Foo::VALUE;
            //  ^^^^^^^^^^ type: u32
    }
}
use a::Trait;
fn foo() {
    let x = a::Foo::VALUE;
         // ^^^^^^^^^^^^^ type: usize
}
"#,
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
fn trait_impl_in_synstructure_const() {
    check_types(
        r#"
struct S;

trait Tr {
    fn method(&self) -> u16;
}

const _DERIVE_Tr_: () = {
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

#[test]
fn resolve_const_generic_array_methods() {
    check_types(
        r#"
#[lang = "array"]
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    pub fn map<F, U>(self, f: F) -> [U; N]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

#[lang = "slice"]
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn map<F, U>(self, f: F) -> &[U]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

fn f() {
    let v = [1, 2].map::<_, usize>(|x| -> x * 2);
    v;
  //^ [usize; 2]
}
    "#,
    );
}

#[test]
fn resolve_const_generic_method() {
    check_types(
        r#"
struct Const<const N: usize>;

#[lang = "array"]
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    pub fn my_map<F, U, const X: usize>(self, f: F, c: Const<X>) -> [U; X]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

#[lang = "slice"]
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn my_map<F, const X: usize, U>(self, f: F, c: Const<X>) -> &[U]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

fn f<const C: usize, P>() {
    let v = [1, 2].my_map::<_, (), 12>(|x| -> x * 2, Const::<12>);
    v;
  //^ [(); 12]
    let v = [1, 2].my_map::<_, P, C>(|x| -> x * 2, Const::<C>);
    v;
  //^ [P; C]
}
    "#,
    );
}

#[test]
fn const_generic_type_alias() {
    check_types(
        r#"
struct Const<const N: usize>;
type U2 = Const<2>;
type U5 = Const<5>;

impl U2 {
    fn f(self) -> Const<12> {
        loop {}
    }
}

impl U5 {
    fn f(self) -> Const<15> {
        loop {}
    }
}

fn f(x: U2) {
    let y = x.f();
      //^ Const<12>
}
    "#,
    );
}

#[test]
fn skip_array_during_method_dispatch() {
    check_types(
        r#"
//- /main2018.rs crate:main2018 deps:core edition:2018
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ &i32

    let a = [0, 1].into_iter();
    a;
  //^ &i32
}

//- /main2021.rs crate:main2021 deps:core edition:2021
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ i32

    let a = [0, 1].into_iter();
    a;
  //^ &i32
}

//- /core.rs crate:core
#[rustc_skip_array_during_method_dispatch]
pub trait IntoIterator {
    type Out;
    fn into_iter(self) -> Self::Out;
}

impl<T> IntoIterator for [T; 1] {
    type Out = T;
    fn into_iter(self) -> Self::Out { loop {} }
}
impl<'a, T> IntoIterator for &'a [T] {
    type Out = &'a T;
    fn into_iter(self) -> Self::Out { loop {} }
}
    "#,
    );
}

#[test]
fn sized_blanket_impl() {
    check_infer(
        r#"
//- minicore: sized
trait Foo { fn foo() -> u8; }
impl<T: Sized> Foo for T {}
fn f<S: Sized, T, U: ?Sized>() {
    u32::foo;
    S::foo;
    T::foo;
    U::foo;
    <[u32]>::foo;
}
"#,
        expect![[r#"
            89..160 '{     ...foo; }': ()
            95..103 'u32::foo': fn foo<u32>() -> u8
            109..115 'S::foo': fn foo<S>() -> u8
            121..127 'T::foo': fn foo<T>() -> u8
            133..139 'U::foo': {unknown}
            145..157 '<[u32]>::foo': {unknown}
        "#]],
    );
}

#[test]
fn local_impl() {
    check_types(
        r#"
fn main() {
    struct SomeStruct(i32);

    impl SomeStruct {
        fn is_even(&self) -> bool {
            self.0 % 2 == 0
        }
    }

    let o = SomeStruct(3);
    let is_even = o.is_even();
     // ^^^^^^^ bool
}
    "#,
    );
}

#[test]
fn deref_fun_1() {
    check_types(
        r#"
//- minicore: deref

struct A<T, U>(T, U);
struct B<T>(T);
struct C<T>(T);

impl<T> core::ops::Deref for A<B<T>, u32> {
    type Target = B<T>;
    fn deref(&self) -> &B<T> { &self.0 }
}
impl core::ops::Deref for B<isize> {
    type Target = C<isize>;
    fn deref(&self) -> &C<isize> { loop {} }
}

impl<T: Copy> C<T> {
    fn thing(&self) -> T { self.0 }
}

fn make<T>() -> T { loop {} }

fn test() {
    let a1 = A(make(), make());
    let _: usize = (*a1).0;
    a1;
  //^^ A<B<usize>, u32>

    let a2 = A(make(), make());
    a2.thing();
  //^^^^^^^^^^ isize
    a2;
  //^^ A<B<isize>, u32>
}
"#,
    );
}

#[test]
fn deref_fun_2() {
    check_types(
        r#"
//- minicore: deref

struct A<T, U>(T, U);
struct B<T>(T);
struct C<T>(T);

impl<T> core::ops::Deref for A<B<T>, u32> {
    type Target = B<T>;
    fn deref(&self) -> &B<T> { &self.0 }
}
impl core::ops::Deref for B<isize> {
    type Target = C<isize>;
    fn deref(&self) -> &C<isize> { loop {} }
}

impl<T> core::ops::Deref for A<C<T>, i32> {
    type Target = C<T>;
    fn deref(&self) -> &C<T> { &self.0 }
}

impl<T: Copy> C<T> {
    fn thing(&self) -> T { self.0 }
}

fn make<T>() -> T { loop {} }

fn test() {
    let a1 = A(make(), 1u32);
    a1.thing();
    a1;
  //^^ A<B<isize>, u32>

    let a2 = A(make(), 1i32);
    let _: &str = a2.thing();
    a2;
  //^^ A<C<&str>, i32>
}
"#,
    );
}

#[test]
fn receiver_adjustment_autoref() {
    check(
        r#"
struct Foo;
impl Foo {
    fn foo(&self) {}
}
fn test() {
    Foo.foo();
  //^^^ adjustments: Borrow(Ref(Not))
    (&Foo).foo();
  // ^^^^ adjustments: Deref(None), Borrow(Ref(Not))
}
"#,
    );
}

#[test]
fn receiver_adjustment_unsize_array() {
    check(
        r#"
//- minicore: slice
fn test() {
    let a = [1, 2, 3];
    a.len();
} //^ adjustments: Borrow(Ref(Not)), Pointer(Unsize)
"#,
    );
}

#[test]
fn bad_inferred_reference_1() {
    check_no_mismatches(
        r#"
//- minicore: sized
pub trait Into<T>: Sized {
    fn into(self) -> T;
}
impl<T> Into<T> for T {
    fn into(self) -> T { self }
}

trait ExactSizeIterator {
    fn len(&self) -> usize;
}

pub struct Foo;
impl Foo {
    fn len(&self) -> usize { 0 }
}

pub fn test(generic_args: impl Into<Foo>) {
    let generic_args = generic_args.into();
    generic_args.len();
    let _: Foo = generic_args;
}
"#,
    );
}

#[test]
fn bad_inferred_reference_2() {
    check_no_mismatches(
        r#"
//- minicore: deref
trait ExactSizeIterator {
    fn len(&self) -> usize;
}

pub struct Foo;
impl Foo {
    fn len(&self) -> usize { 0 }
}

pub fn test() {
    let generic_args;
    generic_args.len();
    let _: Foo = generic_args;
}
"#,
    );
}

#[test]
fn resolve_minicore_iterator() {
    check_types(
        r#"
//- minicore: iterators, sized
fn foo() {
    let m = core::iter::repeat(()).filter_map(|()| Some(92)).next();
}         //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Option<i32>
"#,
    );
}

#[test]
fn primitive_assoc_fn_shadowed_by_use() {
    check_types(
        r#"
//- /lib.rs crate:lib deps:core
use core::u16;

fn f() -> u16 {
    let x = u16::from_le_bytes();
      x
    //^ u16
}

//- /core.rs crate:core
pub mod u16 {}

impl u16 {
    pub fn from_le_bytes() -> Self { 0 }
}
        "#,
    )
}

#[test]
fn with_impl_bounds() {
    check_types(
        r#"
trait Trait {}
struct Foo<T>(T);
impl Trait for isize {}

impl<T: Trait> Foo<T> {
  fn foo() -> isize { 0 }
  fn bar(&self) -> isize { 0 }
}

impl Foo<()> {
  fn foo() {}
  fn bar(&self) {}
}

fn f() {
  let _ = Foo::<isize>::foo();
    //^isize
  let _ = Foo(0isize).bar();
    //^isize
  let _ = Foo::<()>::foo();
    //^()
  let _ = Foo(()).bar();
    //^()
  let _ = Foo::<usize>::foo();
    //^{unknown}
  let _ = Foo(0usize).bar();
    //^{unknown}
}

fn g<T: Trait>(a: T) {
    let _ = Foo::<T>::foo();
      //^isize
    let _ = Foo(a).bar();
      //^isize
}
        "#,
    );
}

#[test]
fn incoherent_impls() {
    check(
        r#"
//- minicore: error, send
pub struct Box<T>(T);
use core::error::Error;

impl dyn Error {
    #[rustc_allow_incoherent_impl]
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<dyn Error>> {
        loop {}
    }
}
impl dyn Error + Send {
    #[rustc_allow_incoherent_impl]
    /// Attempts to downcast the box to a concrete type.
    pub fn downcast<T: Error + 'static>(self: Box<Self>) -> Result<Box<T>, Box<dyn Error + Send>> {
        let err: Box<dyn Error> = self;
                               // ^^^^ expected Box<dyn Error>, got Box<dyn Error + Send>
                               // FIXME, type mismatch should not occur
        <dyn Error>::downcast(err).map_err(|_| loop {})
      //^^^^^^^^^^^^^^^^^^^^^ type: fn downcast<{unknown}>(Box<dyn Error>) -> Result<Box<{unknown}>, Box<dyn Error>>
    }
}
"#,
    );
}

#[test]
fn fallback_private_methods() {
    check(
        r#"
mod module {
    pub struct Struct;

    impl Struct {
        fn func(&self) {}
    }
}

fn foo() {
    let s = module::Struct;
    s.func();
  //^^^^^^^^ type: ()
}
"#,
    );
}

#[test]
fn box_deref_is_builtin() {
    check(
        r#"
//- minicore: deref
use core::ops::Deref;

#[lang = "owned_box"]
struct Box<T>(*mut T);

impl<T> Box<T> {
    fn new(t: T) -> Self {
        loop {}
    }
}

impl<T> Deref for Box<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target;
}

struct Foo;
impl Foo {
    fn foo(&self) {}
}
fn test() {
    Box::new(Foo).foo();
  //^^^^^^^^^^^^^ adjustments: Deref(None), Borrow(Ref(Not))
}
"#,
    );
}

#[test]
fn manually_drop_deref_is_not_builtin() {
    check(
        r#"
//- minicore: manually_drop, deref
struct Foo;
impl Foo {
    fn foo(&self) {}
}
use core::mem::ManuallyDrop;
fn test() {
    ManuallyDrop::new(Foo).foo();
  //^^^^^^^^^^^^^^^^^^^^^^ adjustments: Deref(Some(OverloadedDeref(Some(Not)))), Borrow(Ref(Not))
}
"#,
    );
}
