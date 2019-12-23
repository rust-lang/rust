use super::{infer, type_at, type_at_pos};
use crate::test_db::TestDB;
use insta::assert_snapshot;
use ra_db::fixture::WithFixture;

#[test]
fn infer_slice_method() {
    assert_snapshot!(
        infer(r#"
#[lang = "slice"]
impl<T> [T] {
    fn foo(&self) -> T {
        loop {}
    }
}

#[lang = "slice_alloc"]
impl<T> [T] {}

fn test() {
    <[_]>::foo(b"foo");
}
"#),
        @r###"
    [45; 49) 'self': &[T]
    [56; 79) '{     ...     }': T
    [66; 73) 'loop {}': !
    [71; 73) '{}': ()
    [133; 160) '{     ...o"); }': ()
    [139; 149) '<[_]>::foo': fn foo<u8>(&[T]) -> T
    [139; 157) '<[_]>:..."foo")': u8
    [150; 156) 'b"foo"': &[u8]
    "###
    );
}

#[test]
fn infer_associated_method_struct() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [49; 75) '{     ...     }': A
    [59; 69) 'A { x: 0 }': A
    [66; 67) '0': u32
    [88; 122) '{     ...a.x; }': ()
    [98; 99) 'a': A
    [102; 108) 'A::new': fn new() -> A
    [102; 110) 'A::new()': A
    [116; 117) 'a': A
    [116; 119) 'a.x': u32
    "###
    );
}

#[test]
fn infer_associated_method_enum() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [47; 67) '{     ...     }': A
    [57; 61) 'A::B': A
    [88; 108) '{     ...     }': A
    [98; 102) 'A::C': A
    [121; 178) '{     ...  c; }': ()
    [131; 132) 'a': A
    [135; 139) 'A::b': fn b() -> A
    [135; 141) 'A::b()': A
    [147; 148) 'a': A
    [158; 159) 'c': A
    [162; 166) 'A::c': fn c() -> A
    [162; 168) 'A::c()': A
    [174; 175) 'c': A
    "###
    );
}

#[test]
fn infer_associated_method_with_modules() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [56; 64) '{ A {} }': A
    [58; 62) 'A {}': A
    [126; 132) '{ 99 }': u32
    [128; 130) '99': u32
    [202; 210) '{ C {} }': C
    [204; 208) 'C {}': C
    [241; 325) '{     ...g(); }': ()
    [251; 252) 'x': A
    [255; 266) 'a::A::thing': fn thing() -> A
    [255; 268) 'a::A::thing()': A
    [278; 279) 'y': u32
    [282; 293) 'b::B::thing': fn thing() -> u32
    [282; 295) 'b::B::thing()': u32
    [305; 306) 'z': C
    [309; 320) 'c::C::thing': fn thing() -> C
    [309; 322) 'c::C::thing()': C
    "###
    );
}

#[test]
fn infer_associated_method_generics() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [64; 67) 'val': T
    [82; 109) '{     ...     }': Gen<T>
    [92; 103) 'Gen { val }': Gen<T>
    [98; 101) 'val': T
    [123; 155) '{     ...32); }': ()
    [133; 134) 'a': Gen<u32>
    [137; 146) 'Gen::make': fn make<u32>(T) -> Gen<T>
    [137; 152) 'Gen::make(0u32)': Gen<u32>
    [147; 151) '0u32': u32
    "###
    );
}

#[test]
fn infer_associated_method_generics_with_default_param() {
    assert_snapshot!(
        infer(r#"
struct Gen<T=u32> {
    val: T
}

impl<T> Gen<T> {
    pub fn make() -> Gen<T> {
        loop { }
    }
}

fn test() {
    let a = Gen::make();
}
"#),
        @r###"
    [80; 104) '{     ...     }': Gen<T>
    [90; 98) 'loop { }': !
    [95; 98) '{ }': ()
    [118; 146) '{     ...e(); }': ()
    [128; 129) 'a': Gen<u32>
    [132; 141) 'Gen::make': fn make<u32>() -> Gen<T>
    [132; 143) 'Gen::make()': Gen<u32>
    "###
    );
}

#[test]
fn infer_associated_method_generics_with_default_tuple_param() {
    let t = type_at(
        r#"
//- /main.rs
struct Gen<T=()> {
    val: T
}

impl<T> Gen<T> {
    pub fn make() -> Gen<T> {
        loop { }
    }
}

fn test() {
    let a = Gen::make();
    a.val<|>;
}
"#,
    );
    assert_eq!(t, "()");
}

#[test]
fn infer_associated_method_generics_without_args() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [76; 100) '{     ...     }': Gen<T>
    [86; 94) 'loop { }': !
    [91; 94) '{ }': ()
    [114; 149) '{     ...e(); }': ()
    [124; 125) 'a': Gen<u32>
    [128; 144) 'Gen::<...::make': fn make<u32>() -> Gen<T>
    [128; 146) 'Gen::<...make()': Gen<u32>
    "###
    );
}

#[test]
fn infer_associated_method_generics_2_type_params_without_args() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [102; 126) '{     ...     }': Gen<u32, T>
    [112; 120) 'loop { }': !
    [117; 120) '{ }': ()
    [140; 180) '{     ...e(); }': ()
    [150; 151) 'a': Gen<u32, u64>
    [154; 175) 'Gen::<...::make': fn make<u64>() -> Gen<u32, T>
    [154; 177) 'Gen::<...make()': Gen<u32, u64>
    "###
    );
}

#[test]
fn cross_crate_associated_method_call() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:other_crate
fn test() {
    let x = other_crate::foo::S::thing();
    x<|>;
}

//- /lib.rs crate:other_crate
mod foo {
    struct S;
    impl S {
        fn thing() -> i128 {}
    }
}
"#,
    );
    assert_eq!("i128", type_at_pos(&db, pos));
}

#[test]
fn infer_trait_method_simple() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [31; 35) 'self': &Self
    [110; 114) 'self': &Self
    [170; 228) '{     ...i128 }': ()
    [176; 178) 'S1': S1
    [176; 187) 'S1.method()': u32
    [203; 205) 'S2': S2
    [203; 214) 'S2.method()': i128
    "###
    );
}

#[test]
fn infer_trait_method_scoped() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [63; 67) 'self': &Self
    [169; 173) 'self': &Self
    [300; 337) '{     ...     }': ()
    [310; 311) 'S': S
    [310; 320) 'S.method()': u32
    [416; 454) '{     ...     }': ()
    [426; 427) 'S': S
    [426; 436) 'S.method()': i128
    "###
    );
}

#[test]
fn infer_trait_method_generic_1() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn method(&self) -> T;
}
struct S;
impl Trait<u32> for S {}
fn test() {
    S.method();
}
"#),
        @r###"
    [33; 37) 'self': &Self
    [92; 111) '{     ...d(); }': ()
    [98; 99) 'S': S
    [98; 108) 'S.method()': u32
    "###
    );
}

#[test]
fn infer_trait_method_generic_more_params() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [43; 47) 'self': &Self
    [82; 86) 'self': &Self
    [210; 361) '{     ..., i8 }': ()
    [216; 218) 'S1': S1
    [216; 228) 'S1.method1()': (u8, u16, u32)
    [250; 252) 'S1': S1
    [250; 262) 'S1.method2()': (u32, u16, u8)
    [284; 286) 'S2': S2
    [284; 296) 'S2.method1()': (i8, i16, {unknown})
    [324; 326) 'S2': S2
    [324; 336) 'S2.method2()': ({unknown}, i16, i8)
    "###
    );
}

#[test]
fn infer_trait_method_generic_2() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn method(&self) -> T;
}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn test() {
    S(1u32).method();
}
"#),
        @r###"
    [33; 37) 'self': &Self
    [102; 127) '{     ...d(); }': ()
    [108; 109) 'S': S<u32>(T) -> S<T>
    [108; 115) 'S(1u32)': S<u32>
    [108; 124) 'S(1u32...thod()': u32
    [110; 114) '1u32': u32
    "###
    );
}

#[test]
fn infer_trait_assoc_method() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [87; 193) '{     ...t(); }': ()
    [97; 99) 's1': S
    [105; 121) 'Defaul...efault': fn default<S>() -> Self
    [105; 123) 'Defaul...ault()': S
    [133; 135) 's2': S
    [138; 148) 'S::default': fn default<S>() -> Self
    [138; 150) 'S::default()': S
    [160; 162) 's3': S
    [165; 188) '<S as ...efault': fn default<S>() -> Self
    [165; 190) '<S as ...ault()': S
    "###
    );
}

#[test]
fn infer_trait_assoc_method_generics_1() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [127; 211) '{     ...e(); }': ()
    [137; 138) 'a': u32
    [141; 148) 'S::make': fn make<S, u32>() -> T
    [141; 150) 'S::make()': u32
    [160; 161) 'b': u64
    [164; 178) 'G::<u64>::make': fn make<G<u64>, u64>() -> T
    [164; 180) 'G::<u6...make()': u64
    [190; 191) 'c': f64
    [199; 206) 'G::make': fn make<G<f64>, f64>() -> T
    [199; 208) 'G::make()': f64
    "###
    );
}

#[test]
fn infer_trait_assoc_method_generics_2() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [135; 313) '{     ...e(); }': ()
    [145; 146) 'a': (u32, i64)
    [149; 163) 'S::make::<i64>': fn make<S, u32, i64>() -> (T, U)
    [149; 165) 'S::mak...i64>()': (u32, i64)
    [175; 176) 'b': (u32, i64)
    [189; 196) 'S::make': fn make<S, u32, i64>() -> (T, U)
    [189; 198) 'S::make()': (u32, i64)
    [208; 209) 'c': (u32, i64)
    [212; 233) 'G::<u3...:<i64>': fn make<G<u32>, u32, i64>() -> (T, U)
    [212; 235) 'G::<u3...i64>()': (u32, i64)
    [245; 246) 'd': (u32, i64)
    [259; 273) 'G::make::<i64>': fn make<G<u32>, u32, i64>() -> (T, U)
    [259; 275) 'G::mak...i64>()': (u32, i64)
    [285; 286) 'e': (u32, i64)
    [301; 308) 'G::make': fn make<G<u32>, u32, i64>() -> (T, U)
    [301; 310) 'G::make()': (u32, i64)
    "###
    );
}

#[test]
fn infer_trait_assoc_method_generics_3() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn make() -> (Self, T);
}
struct S<T>;
impl Trait<i64> for S<i32> {}
fn test() {
    let a = S::make();
}
"#),
        @r###"
    [101; 127) '{     ...e(); }': ()
    [111; 112) 'a': (S<i32>, i64)
    [115; 122) 'S::make': fn make<S<i32>, i64>() -> (Self, T)
    [115; 124) 'S::make()': (S<i32>, i64)
    "###
    );
}

#[test]
fn infer_trait_assoc_method_generics_4() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [131; 203) '{     ...e(); }': ()
    [141; 142) 'a': (S<u64>, i64)
    [158; 165) 'S::make': fn make<S<u64>, i64>() -> (Self, T)
    [158; 167) 'S::make()': (S<u64>, i64)
    [177; 178) 'b': (S<u32>, i32)
    [191; 198) 'S::make': fn make<S<u32>, i32>() -> (Self, T)
    [191; 200) 'S::make()': (S<u32>, i32)
    "###
    );
}

#[test]
fn infer_trait_assoc_method_generics_5() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn make<U>() -> (Self, T, U);
}
struct S<T>;
impl Trait<i64> for S<u64> {}
fn test() {
    let a = <S as Trait<i64>>::make::<u8>();
    let b: (S<u64>, _, _) = Trait::<i64>::make::<u8>();
}
"#),
        @r###"
    [107; 211) '{     ...>(); }': ()
    [117; 118) 'a': (S<u64>, i64, u8)
    [121; 150) '<S as ...::<u8>': fn make<S<u64>, i64, u8>() -> (Self, T, U)
    [121; 152) '<S as ...<u8>()': (S<u64>, i64, u8)
    [162; 163) 'b': (S<u64>, i64, u8)
    [182; 206) 'Trait:...::<u8>': fn make<S<u64>, i64, u8>() -> (Self, T, U)
    [182; 208) 'Trait:...<u8>()': (S<u64>, i64, u8)
    "###
    );
}

#[test]
fn infer_call_trait_method_on_generic_param_1() {
    assert_snapshot!(
        infer(r#"
trait Trait {
    fn method(&self) -> u32;
}
fn test<T: Trait>(t: T) {
    t.method();
}
"#),
        @r###"
    [30; 34) 'self': &Self
    [64; 65) 't': T
    [70; 89) '{     ...d(); }': ()
    [76; 77) 't': T
    [76; 86) 't.method()': u32
    "###
    );
}

#[test]
fn infer_call_trait_method_on_generic_param_2() {
    assert_snapshot!(
        infer(r#"
trait Trait<T> {
    fn method(&self) -> T;
}
fn test<U, T: Trait<U>>(t: T) {
    t.method();
}
"#),
        @r###"
    [33; 37) 'self': &Self
    [71; 72) 't': T
    [77; 96) '{     ...d(); }': ()
    [83; 84) 't': T
    [83; 93) 't.method()': [missing name]
    "###
    );
}

#[test]
fn infer_with_multiple_trait_impls() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    [29; 33) 'self': Self
    [111; 202) '{     ...(S); }': ()
    [121; 122) 'x': u32
    [130; 131) 'S': S
    [130; 138) 'S.into()': u32
    [148; 149) 'y': u64
    [157; 158) 'S': S
    [157; 165) 'S.into()': u64
    [175; 176) 'z': u64
    [179; 196) 'Into::...::into': fn into<S, u64>(Self) -> T
    [179; 199) 'Into::...nto(S)': u64
    [197; 198) 'S': S
    "###
    );
}

#[test]
fn method_resolution_unify_impl_self_type() {
    let t = type_at(
        r#"
//- /main.rs
struct S<T>;
impl S<u32> { fn foo(&self) -> u8 {} }
impl S<i32> { fn foo(&self) -> i8 {} }
fn test() { (S::<u32>.foo(), S::<i32>.foo())<|>; }
"#,
    );
    assert_eq!(t, "(u8, i8)");
}

#[test]
fn method_resolution_trait_before_autoref() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(&self) -> i8 { 0 } }
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn method_resolution_by_value_before_autoref() {
    let t = type_at(
        r#"
//- /main.rs
trait Clone { fn clone(&self) -> Self; }
struct S;
impl Clone for S {}
impl Clone for &S {}
fn test() { (S.clone(), (&S).clone(), (&&S).clone())<|>; }
"#,
    );
    assert_eq!(t, "(S, S, &S)");
}

#[test]
fn method_resolution_trait_before_autoderef() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(self) -> i8 { 0 } }
impl Trait for &S { fn foo(self) -> u128 { 0 } }
fn test() { (&S).foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn method_resolution_impl_before_trait() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(self) -> i8 { 0 } }
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "i8");
}

#[test]
fn method_resolution_impl_ref_before_trait() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl S { fn foo(&self) -> i8 { 0 } }
impl Trait for &S { fn foo(self) -> u128 { 0 } }
fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "i8");
}

#[test]
fn method_resolution_trait_autoderef() {
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { (&S).foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn method_resolution_trait_from_prelude() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:other_crate
struct S;
impl Clone for S {}

fn test() {
    S.clone()<|>;
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
    assert_eq!("S", type_at_pos(&db, pos));
}

#[test]
fn method_resolution_where_clause_for_unknown_trait() {
    // The blanket impl currently applies because we ignore the unresolved where clause
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T> Trait for T where T: UnknownTrait {}
fn test() { (&S).foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn method_resolution_where_clause_not_met() {
    // The blanket impl shouldn't apply because we can't prove S: Clone
    let t = type_at(
        r#"
//- /main.rs
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T> Trait for T where T: Clone {}
fn test() { (&S).foo()<|>; }
"#,
    );
    // This is also to make sure that we don't resolve to the foo method just
    // because that's the only method named foo we can find, which would make
    // the below tests not work
    assert_eq!(t, "{unknown}");
}

#[test]
fn method_resolution_where_clause_inline_not_met() {
    // The blanket impl shouldn't apply because we can't prove S: Clone
    let t = type_at(
        r#"
//- /main.rs
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T: Clone> Trait for T {}
fn test() { (&S).foo()<|>; }
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn method_resolution_where_clause_1() {
    let t = type_at(
        r#"
//- /main.rs
trait Clone {}
trait Trait { fn foo(self) -> u128; }
struct S;
impl Clone for S {}
impl<T> Trait for T where T: Clone {}
fn test() { S.foo()<|>; }
"#,
    );
    assert_eq!(t, "u128");
}

#[test]
fn method_resolution_where_clause_2() {
    let t = type_at(
        r#"
//- /main.rs
trait Into<T> { fn into(self) -> T; }
trait From<T> { fn from(other: T) -> Self; }
struct S1;
struct S2;
impl From<S2> for S1 {}
impl<T, U> Into<U> for T where U: From<T> {}
fn test() { S2.into()<|>; }
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn method_resolution_where_clause_inline() {
    let t = type_at(
        r#"
//- /main.rs
trait Into<T> { fn into(self) -> T; }
trait From<T> { fn from(other: T) -> Self; }
struct S1;
struct S2;
impl From<S2> for S1 {}
impl<T, U: From<T>> Into<U> for T {}
fn test() { S2.into()<|>; }
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn method_resolution_encountering_fn_type() {
    type_at(
        r#"
//- /main.rs
fn foo() {}
trait FnOnce { fn call(self); }
fn test() { foo.call()<|>; }
"#,
    );
}

#[test]
fn method_resolution_slow() {
    // this can get quite slow if we set the solver size limit too high
    let t = type_at(
        r#"
//- /main.rs
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

fn test() { (S {}).method()<|>; }
"#,
    );
    assert_eq!(t, "()");
}
