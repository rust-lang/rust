use super::{infer, type_at, type_at_pos};
use crate::test_db::TestDB;
use insta::assert_snapshot;
use ra_db::fixture::WithFixture;

#[test]
fn infer_box() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std

fn test() {
    let x = box 1;
    let t = (x, box x, box &1, box [1]);
    t<|>;
}

//- /std.rs crate:std
#[prelude_import] use prelude::*;
mod prelude {}

mod boxed {
    #[lang = "owned_box"]
    pub struct Box<T: ?Sized> {
        inner: *mut T,
    }
}

"#,
    );
    assert_eq!("(Box<i32>, Box<Box<i32>>, Box<&i32>, Box<[i32;_]>)", type_at_pos(&db, pos));
}

#[test]
fn infer_adt_self() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs
enum Nat { Succ(Self), Demo(Nat), Zero }

fn test() {
    let foo: Nat = Nat::Zero;
    if let Nat::Succ(x) = foo {
        x<|>
    }
}

"#,
    );
    assert_eq!("Nat", type_at_pos(&db, pos));
}

#[test]
fn infer_ranges() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs crate:main deps:std
fn test() {
    let a = ..;
    let b = 1..;
    let c = ..2u32;
    let d = 1..2usize;
    let e = ..=10;
    let f = 'a'..='z';

    let t = (a, b, c, d, e, f);
    t<|>;
}

//- /std.rs crate:std
#[prelude_import] use prelude::*;
mod prelude {}

pub mod ops {
    pub struct Range<Idx> {
        pub start: Idx,
        pub end: Idx,
    }
    pub struct RangeFrom<Idx> {
        pub start: Idx,
    }
    struct RangeFull;
    pub struct RangeInclusive<Idx> {
        start: Idx,
        end: Idx,
        is_empty: u8,
    }
    pub struct RangeTo<Idx> {
        pub end: Idx,
    }
    pub struct RangeToInclusive<Idx> {
        pub end: Idx,
    }
}
"#,
    );
    assert_eq!(
        "(RangeFull, RangeFrom<i32>, RangeTo<u32>, Range<usize>, RangeToInclusive<i32>, RangeInclusive<char>)",
        type_at_pos(&db, pos),
    );
}

#[test]
fn infer_while_let() {
    let (db, pos) = TestDB::with_position(
        r#"
//- /main.rs
enum Option<T> { Some(T), None }

fn test() {
    let foo: Option<f32> = None;
    while let Option::Some(x) = foo {
        <|>x
    }
}

"#,
    );
    assert_eq!("f32", type_at_pos(&db, pos));
}

#[test]
fn infer_basics() {
    assert_snapshot!(
        infer(r#"
fn test(a: u32, b: isize, c: !, d: &str) {
    a;
    b;
    c;
    d;
    1usize;
    1isize;
    "test";
    1.0f32;
}"#),
        @r###"
    [9; 10) 'a': u32
    [17; 18) 'b': isize
    [27; 28) 'c': !
    [33; 34) 'd': &str
    [42; 121) '{     ...f32; }': !
    [48; 49) 'a': u32
    [55; 56) 'b': isize
    [62; 63) 'c': !
    [69; 70) 'd': &str
    [76; 82) '1usize': usize
    [88; 94) '1isize': isize
    [100; 106) '"test"': &str
    [112; 118) '1.0f32': f32
    "###
    );
}

#[test]
fn infer_let() {
    assert_snapshot!(
        infer(r#"
fn test() {
    let a = 1isize;
    let b: usize = 1;
    let c = b;
    let d: u32;
    let e;
    let f: i32 = e;
}
"#),
        @r###"
    [11; 118) '{     ...= e; }': ()
    [21; 22) 'a': isize
    [25; 31) '1isize': isize
    [41; 42) 'b': usize
    [52; 53) '1': usize
    [63; 64) 'c': usize
    [67; 68) 'b': usize
    [78; 79) 'd': u32
    [94; 95) 'e': i32
    [105; 106) 'f': i32
    [114; 115) 'e': i32
    "###
    );
}

#[test]
fn infer_paths() {
    assert_snapshot!(
        infer(r#"
fn a() -> u32 { 1 }

mod b {
    fn c() -> u32 { 1 }
}

fn test() {
    a();
    b::c();
}
"#),
        @r###"
    [15; 20) '{ 1 }': u32
    [17; 18) '1': u32
    [48; 53) '{ 1 }': u32
    [50; 51) '1': u32
    [67; 91) '{     ...c(); }': ()
    [73; 74) 'a': fn a() -> u32
    [73; 76) 'a()': u32
    [82; 86) 'b::c': fn c() -> u32
    [82; 88) 'b::c()': u32
    "###
    );
}

#[test]
fn infer_path_type() {
    assert_snapshot!(
        infer(r#"
struct S;

impl S {
    fn foo() -> i32 { 1 }
}

fn test() {
    S::foo();
    <S>::foo();
}
"#),
        @r###"
    [41; 46) '{ 1 }': i32
    [43; 44) '1': i32
    [60; 93) '{     ...o(); }': ()
    [66; 72) 'S::foo': fn foo() -> i32
    [66; 74) 'S::foo()': i32
    [80; 88) '<S>::foo': fn foo() -> i32
    [80; 90) '<S>::foo()': i32
    "###
    );
}

#[test]
fn infer_struct() {
    assert_snapshot!(
        infer(r#"
struct A {
    b: B,
    c: C,
}
struct B;
struct C(usize);

fn test() {
    let c = C(1);
    B;
    let a: A = A { b: B, c: C(1) };
    a.b;
    a.c;
}
"#),
        @r###"
    [72; 154) '{     ...a.c; }': ()
    [82; 83) 'c': C
    [86; 87) 'C': C(usize) -> C
    [86; 90) 'C(1)': C
    [88; 89) '1': usize
    [96; 97) 'B': B
    [107; 108) 'a': A
    [114; 133) 'A { b:...C(1) }': A
    [121; 122) 'B': B
    [127; 128) 'C': C(usize) -> C
    [127; 131) 'C(1)': C
    [129; 130) '1': usize
    [139; 140) 'a': A
    [139; 142) 'a.b': B
    [148; 149) 'a': A
    [148; 151) 'a.c': C
    "###
    );
}

#[test]
fn infer_enum() {
    assert_snapshot!(
        infer(r#"
enum E {
  V1 { field: u32 },
  V2
}
fn test() {
  E::V1 { field: 1 };
  E::V2;
}"#),
        @r###"
    [48; 82) '{   E:...:V2; }': ()
    [52; 70) 'E::V1 ...d: 1 }': E
    [67; 68) '1': u32
    [74; 79) 'E::V2': E
    "###
    );
}

#[test]
fn infer_refs() {
    assert_snapshot!(
        infer(r#"
fn test(a: &u32, b: &mut u32, c: *const u32, d: *mut u32) {
    a;
    *a;
    &a;
    &mut a;
    b;
    *b;
    &b;
    c;
    *c;
    d;
    *d;
}
"#),
        @r###"
    [9; 10) 'a': &u32
    [18; 19) 'b': &mut u32
    [31; 32) 'c': *const u32
    [46; 47) 'd': *mut u32
    [59; 150) '{     ... *d; }': ()
    [65; 66) 'a': &u32
    [72; 74) '*a': u32
    [73; 74) 'a': &u32
    [80; 82) '&a': &&u32
    [81; 82) 'a': &u32
    [88; 94) '&mut a': &mut &u32
    [93; 94) 'a': &u32
    [100; 101) 'b': &mut u32
    [107; 109) '*b': u32
    [108; 109) 'b': &mut u32
    [115; 117) '&b': &&mut u32
    [116; 117) 'b': &mut u32
    [123; 124) 'c': *const u32
    [130; 132) '*c': u32
    [131; 132) 'c': *const u32
    [138; 139) 'd': *mut u32
    [145; 147) '*d': u32
    [146; 147) 'd': *mut u32
    "###
    );
}

#[test]
fn infer_literals() {
    assert_snapshot!(
        infer(r##"
fn test() {
    5i32;
    5f32;
    5f64;
    "hello";
    b"bytes";
    'c';
    b'b';
    3.14;
    5000;
    false;
    true;
    r#"
        //! doc
        // non-doc
        mod foo {}
        "#;
    br#"yolo"#;
}
"##),
        @r###"
    [11; 221) '{     ...o"#; }': ()
    [17; 21) '5i32': i32
    [27; 31) '5f32': f32
    [37; 41) '5f64': f64
    [47; 54) '"hello"': &str
    [60; 68) 'b"bytes"': &[u8]
    [74; 77) ''c'': char
    [83; 87) 'b'b'': u8
    [93; 97) '3.14': f64
    [103; 107) '5000': i32
    [113; 118) 'false': bool
    [124; 128) 'true': bool
    [134; 202) 'r#"   ...    "#': &str
    [208; 218) 'br#"yolo"#': &[u8]
    "###
    );
}

#[test]
fn infer_unary_op() {
    assert_snapshot!(
        infer(r#"
enum SomeType {}

fn test(x: SomeType) {
    let b = false;
    let c = !b;
    let a = 100;
    let d: i128 = -a;
    let e = -100;
    let f = !!!true;
    let g = !42;
    let h = !10u32;
    let j = !a;
    -3.14;
    !3;
    -x;
    !x;
    -"hello";
    !"hello";
}
"#),
        @r###"
    [27; 28) 'x': SomeType
    [40; 272) '{     ...lo"; }': ()
    [50; 51) 'b': bool
    [54; 59) 'false': bool
    [69; 70) 'c': bool
    [73; 75) '!b': bool
    [74; 75) 'b': bool
    [85; 86) 'a': i128
    [89; 92) '100': i128
    [102; 103) 'd': i128
    [112; 114) '-a': i128
    [113; 114) 'a': i128
    [124; 125) 'e': i32
    [128; 132) '-100': i32
    [129; 132) '100': i32
    [142; 143) 'f': bool
    [146; 153) '!!!true': bool
    [147; 153) '!!true': bool
    [148; 153) '!true': bool
    [149; 153) 'true': bool
    [163; 164) 'g': i32
    [167; 170) '!42': i32
    [168; 170) '42': i32
    [180; 181) 'h': u32
    [184; 190) '!10u32': u32
    [185; 190) '10u32': u32
    [200; 201) 'j': i128
    [204; 206) '!a': i128
    [205; 206) 'a': i128
    [212; 217) '-3.14': f64
    [213; 217) '3.14': f64
    [223; 225) '!3': i32
    [224; 225) '3': i32
    [231; 233) '-x': {unknown}
    [232; 233) 'x': SomeType
    [239; 241) '!x': {unknown}
    [240; 241) 'x': SomeType
    [247; 255) '-"hello"': {unknown}
    [248; 255) '"hello"': &str
    [261; 269) '!"hello"': {unknown}
    [262; 269) '"hello"': &str
    "###
    );
}

#[test]
fn infer_backwards() {
    assert_snapshot!(
        infer(r#"
fn takes_u32(x: u32) {}

struct S { i32_field: i32 }

fn test() -> &mut &f64 {
    let a = unknown_function();
    takes_u32(a);
    let b = unknown_function();
    S { i32_field: b };
    let c = unknown_function();
    &mut &c
}
"#),
        @r###"
    [14; 15) 'x': u32
    [22; 24) '{}': ()
    [78; 231) '{     ...t &c }': &mut &f64
    [88; 89) 'a': u32
    [92; 108) 'unknow...nction': {unknown}
    [92; 110) 'unknow...tion()': u32
    [116; 125) 'takes_u32': fn takes_u32(u32) -> ()
    [116; 128) 'takes_u32(a)': ()
    [126; 127) 'a': u32
    [138; 139) 'b': i32
    [142; 158) 'unknow...nction': {unknown}
    [142; 160) 'unknow...tion()': i32
    [166; 184) 'S { i3...d: b }': S
    [181; 182) 'b': i32
    [194; 195) 'c': f64
    [198; 214) 'unknow...nction': {unknown}
    [198; 216) 'unknow...tion()': f64
    [222; 229) '&mut &c': &mut &f64
    [227; 229) '&c': &f64
    [228; 229) 'c': f64
    "###
    );
}

#[test]
fn infer_self() {
    assert_snapshot!(
        infer(r#"
struct S;

impl S {
    fn test(&self) {
        self;
    }
    fn test2(self: &Self) {
        self;
    }
    fn test3() -> Self {
        S {}
    }
    fn test4() -> Self {
        Self {}
    }
}
"#),
        @r###"
    [34; 38) 'self': &S
    [40; 61) '{     ...     }': ()
    [50; 54) 'self': &S
    [75; 79) 'self': &S
    [88; 109) '{     ...     }': ()
    [98; 102) 'self': &S
    [133; 153) '{     ...     }': S
    [143; 147) 'S {}': S
    [177; 200) '{     ...     }': S
    [187; 194) 'Self {}': S
    "###
    );
}

#[test]
fn infer_binary_op() {
    assert_snapshot!(
        infer(r#"
fn f(x: bool) -> i32 {
    0i32
}

fn test() -> bool {
    let x = a && b;
    let y = true || false;
    let z = x == y;
    let t = x != y;
    let minus_forty: isize = -40isize;
    let h = minus_forty <= CONST_2;
    let c = f(z || y) + 5;
    let d = b;
    let g = minus_forty ^= i;
    let ten: usize = 10;
    let ten_is_eleven = ten == some_num;

    ten < 3
}
"#),
        @r###"
    [6; 7) 'x': bool
    [22; 34) '{     0i32 }': i32
    [28; 32) '0i32': i32
    [54; 370) '{     ... < 3 }': bool
    [64; 65) 'x': bool
    [68; 69) 'a': bool
    [68; 74) 'a && b': bool
    [73; 74) 'b': bool
    [84; 85) 'y': bool
    [88; 92) 'true': bool
    [88; 101) 'true || false': bool
    [96; 101) 'false': bool
    [111; 112) 'z': bool
    [115; 116) 'x': bool
    [115; 121) 'x == y': bool
    [120; 121) 'y': bool
    [131; 132) 't': bool
    [135; 136) 'x': bool
    [135; 141) 'x != y': bool
    [140; 141) 'y': bool
    [151; 162) 'minus_forty': isize
    [172; 180) '-40isize': isize
    [173; 180) '40isize': isize
    [190; 191) 'h': bool
    [194; 205) 'minus_forty': isize
    [194; 216) 'minus_...ONST_2': bool
    [209; 216) 'CONST_2': isize
    [226; 227) 'c': i32
    [230; 231) 'f': fn f(bool) -> i32
    [230; 239) 'f(z || y)': i32
    [230; 243) 'f(z || y) + 5': i32
    [232; 233) 'z': bool
    [232; 238) 'z || y': bool
    [237; 238) 'y': bool
    [242; 243) '5': i32
    [253; 254) 'd': {unknown}
    [257; 258) 'b': {unknown}
    [268; 269) 'g': ()
    [272; 283) 'minus_forty': isize
    [272; 288) 'minus_...y ^= i': ()
    [287; 288) 'i': isize
    [298; 301) 'ten': usize
    [311; 313) '10': usize
    [323; 336) 'ten_is_eleven': bool
    [339; 342) 'ten': usize
    [339; 354) 'ten == some_num': bool
    [346; 354) 'some_num': usize
    [361; 364) 'ten': usize
    [361; 368) 'ten < 3': bool
    [367; 368) '3': usize
    "###
    );
}

#[test]
fn infer_field_autoderef() {
    assert_snapshot!(
        infer(r#"
struct A {
    b: B,
}
struct B;

fn test1(a: A) {
    let a1 = a;
    a1.b;
    let a2 = &a;
    a2.b;
    let a3 = &mut a;
    a3.b;
    let a4 = &&&&&&&a;
    a4.b;
    let a5 = &mut &&mut &&mut a;
    a5.b;
}

fn test2(a1: *const A, a2: *mut A) {
    a1.b;
    a2.b;
}
"#),
        @r###"
    [44; 45) 'a': A
    [50; 213) '{     ...5.b; }': ()
    [60; 62) 'a1': A
    [65; 66) 'a': A
    [72; 74) 'a1': A
    [72; 76) 'a1.b': B
    [86; 88) 'a2': &A
    [91; 93) '&a': &A
    [92; 93) 'a': A
    [99; 101) 'a2': &A
    [99; 103) 'a2.b': B
    [113; 115) 'a3': &mut A
    [118; 124) '&mut a': &mut A
    [123; 124) 'a': A
    [130; 132) 'a3': &mut A
    [130; 134) 'a3.b': B
    [144; 146) 'a4': &&&&&&&A
    [149; 157) '&&&&&&&a': &&&&&&&A
    [150; 157) '&&&&&&a': &&&&&&A
    [151; 157) '&&&&&a': &&&&&A
    [152; 157) '&&&&a': &&&&A
    [153; 157) '&&&a': &&&A
    [154; 157) '&&a': &&A
    [155; 157) '&a': &A
    [156; 157) 'a': A
    [163; 165) 'a4': &&&&&&&A
    [163; 167) 'a4.b': B
    [177; 179) 'a5': &mut &&mut &&mut A
    [182; 200) '&mut &...&mut a': &mut &&mut &&mut A
    [187; 200) '&&mut &&mut a': &&mut &&mut A
    [188; 200) '&mut &&mut a': &mut &&mut A
    [193; 200) '&&mut a': &&mut A
    [194; 200) '&mut a': &mut A
    [199; 200) 'a': A
    [206; 208) 'a5': &mut &&mut &&mut A
    [206; 210) 'a5.b': B
    [224; 226) 'a1': *const A
    [238; 240) 'a2': *mut A
    [250; 273) '{     ...2.b; }': ()
    [256; 258) 'a1': *const A
    [256; 260) 'a1.b': B
    [266; 268) 'a2': *mut A
    [266; 270) 'a2.b': B
    "###
    );
}

#[test]
fn infer_argument_autoderef() {
    assert_snapshot!(
        infer(r#"
#[lang = "deref"]
pub trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct A<T>(T);

impl<T> A<T> {
    fn foo(&self) -> &T {
        &self.0
    }
}

struct B<T>(T);

impl<T> Deref for B<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn test() {
    let t = A::foo(&&B(B(A(42))));
}
"#),
        @r###"
    [68; 72) 'self': &Self
    [139; 143) 'self': &A<T>
    [151; 174) '{     ...     }': &T
    [161; 168) '&self.0': &T
    [162; 166) 'self': &A<T>
    [162; 168) 'self.0': T
    [255; 259) 'self': &B<T>
    [278; 301) '{     ...     }': &T
    [288; 295) '&self.0': &T
    [289; 293) 'self': &B<T>
    [289; 295) 'self.0': T
    [315; 353) '{     ...))); }': ()
    [325; 326) 't': &i32
    [329; 335) 'A::foo': fn foo<i32>(&A<T>) -> &T
    [329; 350) 'A::foo...42))))': &i32
    [336; 349) '&&B(B(A(42)))': &&B<B<A<i32>>>
    [337; 349) '&B(B(A(42)))': &B<B<A<i32>>>
    [338; 339) 'B': B<B<A<i32>>>(T) -> B<T>
    [338; 349) 'B(B(A(42)))': B<B<A<i32>>>
    [340; 341) 'B': B<A<i32>>(T) -> B<T>
    [340; 348) 'B(A(42))': B<A<i32>>
    [342; 343) 'A': A<i32>(T) -> A<T>
    [342; 347) 'A(42)': A<i32>
    [344; 346) '42': i32
    "###
    );
}

#[test]
fn infer_method_argument_autoderef() {
    assert_snapshot!(
        infer(r#"
#[lang = "deref"]
pub trait Deref {
    type Target;
    fn deref(&self) -> &Self::Target;
}

struct A<T>(*mut T);

impl<T> A<T> {
    fn foo(&self, x: &A<T>) -> &T {
        &*x.0
    }
}

struct B<T>(T);

impl<T> Deref for B<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn test(a: A<i32>) {
    let t = A(0 as *mut _).foo(&&B(B(a)));
}
"#),
        @r###"
    [68; 72) 'self': &Self
    [144; 148) 'self': &A<T>
    [150; 151) 'x': &A<T>
    [166; 187) '{     ...     }': &T
    [176; 181) '&*x.0': &T
    [177; 181) '*x.0': T
    [178; 179) 'x': &A<T>
    [178; 181) 'x.0': *mut T
    [268; 272) 'self': &B<T>
    [291; 314) '{     ...     }': &T
    [301; 308) '&self.0': &T
    [302; 306) 'self': &B<T>
    [302; 308) 'self.0': T
    [326; 327) 'a': A<i32>
    [337; 383) '{     ...))); }': ()
    [347; 348) 't': &i32
    [351; 352) 'A': A<i32>(*mut T) -> A<T>
    [351; 365) 'A(0 as *mut _)': A<i32>
    [351; 380) 'A(0 as...B(a)))': &i32
    [353; 354) '0': i32
    [353; 364) '0 as *mut _': *mut i32
    [370; 379) '&&B(B(a))': &&B<B<A<i32>>>
    [371; 379) '&B(B(a))': &B<B<A<i32>>>
    [372; 373) 'B': B<B<A<i32>>>(T) -> B<T>
    [372; 379) 'B(B(a))': B<B<A<i32>>>
    [374; 375) 'B': B<A<i32>>(T) -> B<T>
    [374; 378) 'B(a)': B<A<i32>>
    [376; 377) 'a': A<i32>
    "###
    );
}

#[test]
fn infer_in_elseif() {
    assert_snapshot!(
        infer(r#"
struct Foo { field: i32 }
fn main(foo: Foo) {
    if true {

    } else if false {
        foo.field
    }
}
"#),
        @r###"
    [35; 38) 'foo': Foo
    [45; 109) '{     ...   } }': ()
    [51; 107) 'if tru...     }': ()
    [54; 58) 'true': bool
    [59; 67) '{      }': ()
    [73; 107) 'if fal...     }': ()
    [76; 81) 'false': bool
    [82; 107) '{     ...     }': i32
    [92; 95) 'foo': Foo
    [92; 101) 'foo.field': i32
    "###
    )
}

#[test]
fn infer_if_match_with_return() {
    assert_snapshot!(
        infer(r#"
fn foo() {
    let _x1 = if true {
        1
    } else {
        return;
    };
    let _x2 = if true {
        2
    } else {
        return
    };
    let _x3 = match true {
        true => 3,
        _ => {
            return;
        }
    };
    let _x4 = match true {
        true => 4,
        _ => return
    };
}"#),
        @r###"
    [10; 323) '{     ...  }; }': ()
    [20; 23) '_x1': i32
    [26; 80) 'if tru...     }': i32
    [29; 33) 'true': bool
    [34; 51) '{     ...     }': i32
    [44; 45) '1': i32
    [57; 80) '{     ...     }': !
    [67; 73) 'return': !
    [90; 93) '_x2': i32
    [96; 149) 'if tru...     }': i32
    [99; 103) 'true': bool
    [104; 121) '{     ...     }': i32
    [114; 115) '2': i32
    [127; 149) '{     ...     }': !
    [137; 143) 'return': !
    [159; 162) '_x3': i32
    [165; 247) 'match ...     }': i32
    [171; 175) 'true': bool
    [186; 190) 'true': bool
    [194; 195) '3': i32
    [205; 206) '_': bool
    [210; 241) '{     ...     }': !
    [224; 230) 'return': !
    [257; 260) '_x4': i32
    [263; 320) 'match ...     }': i32
    [269; 273) 'true': bool
    [284; 288) 'true': bool
    [292; 293) '4': i32
    [303; 304) '_': bool
    [308; 314) 'return': !
    "###
    )
}

#[test]
fn infer_inherent_method() {
    assert_snapshot!(
        infer(r#"
struct A;

impl A {
    fn foo(self, x: u32) -> i32 {}
}

mod b {
    impl super::A {
        fn bar(&self, x: u64) -> i64 {}
    }
}

fn test(a: A) {
    a.foo(1);
    (&a).bar(1);
    a.bar(1);
}
"#),
        @r###"
    [32; 36) 'self': A
    [38; 39) 'x': u32
    [53; 55) '{}': ()
    [103; 107) 'self': &A
    [109; 110) 'x': u64
    [124; 126) '{}': ()
    [144; 145) 'a': A
    [150; 198) '{     ...(1); }': ()
    [156; 157) 'a': A
    [156; 164) 'a.foo(1)': i32
    [162; 163) '1': u32
    [170; 181) '(&a).bar(1)': i64
    [171; 173) '&a': &A
    [172; 173) 'a': A
    [179; 180) '1': u64
    [187; 188) 'a': A
    [187; 195) 'a.bar(1)': i64
    [193; 194) '1': u64
    "###
    );
}

#[test]
fn infer_inherent_method_str() {
    assert_snapshot!(
        infer(r#"
#[lang = "str"]
impl str {
    fn foo(&self) -> i32 {}
}

fn test() {
    "foo".foo();
}
"#),
        @r###"
    [40; 44) 'self': &str
    [53; 55) '{}': ()
    [69; 89) '{     ...o(); }': ()
    [75; 80) '"foo"': &str
    [75; 86) '"foo".foo()': i32
    "###
    );
}

#[test]
fn infer_tuple() {
    assert_snapshot!(
        infer(r#"
fn test(x: &str, y: isize) {
    let a: (u32, &str) = (1, "a");
    let b = (a, x);
    let c = (y, x);
    let d = (c, x);
    let e = (1, "e");
    let f = (e, "d");
}
"#),
        @r###"
    [9; 10) 'x': &str
    [18; 19) 'y': isize
    [28; 170) '{     ...d"); }': ()
    [38; 39) 'a': (u32, &str)
    [55; 63) '(1, "a")': (u32, &str)
    [56; 57) '1': u32
    [59; 62) '"a"': &str
    [73; 74) 'b': ((u32, &str), &str)
    [77; 83) '(a, x)': ((u32, &str), &str)
    [78; 79) 'a': (u32, &str)
    [81; 82) 'x': &str
    [93; 94) 'c': (isize, &str)
    [97; 103) '(y, x)': (isize, &str)
    [98; 99) 'y': isize
    [101; 102) 'x': &str
    [113; 114) 'd': ((isize, &str), &str)
    [117; 123) '(c, x)': ((isize, &str), &str)
    [118; 119) 'c': (isize, &str)
    [121; 122) 'x': &str
    [133; 134) 'e': (i32, &str)
    [137; 145) '(1, "e")': (i32, &str)
    [138; 139) '1': i32
    [141; 144) '"e"': &str
    [155; 156) 'f': ((i32, &str), &str)
    [159; 167) '(e, "d")': ((i32, &str), &str)
    [160; 161) 'e': (i32, &str)
    [163; 166) '"d"': &str
    "###
    );
}

#[test]
fn infer_array() {
    assert_snapshot!(
        infer(r#"
fn test(x: &str, y: isize) {
    let a = [x];
    let b = [a, a];
    let c = [b, b];

    let d = [y, 1, 2, 3];
    let d = [1, y, 2, 3];
    let e = [y];
    let f = [d, d];
    let g = [e, e];

    let h = [1, 2];
    let i = ["a", "b"];

    let b = [a, ["b"]];
    let x: [u8; 0] = [];
}
"#),
        @r###"
    [9; 10) 'x': &str
    [18; 19) 'y': isize
    [28; 293) '{     ... []; }': ()
    [38; 39) 'a': [&str;_]
    [42; 45) '[x]': [&str;_]
    [43; 44) 'x': &str
    [55; 56) 'b': [[&str;_];_]
    [59; 65) '[a, a]': [[&str;_];_]
    [60; 61) 'a': [&str;_]
    [63; 64) 'a': [&str;_]
    [75; 76) 'c': [[[&str;_];_];_]
    [79; 85) '[b, b]': [[[&str;_];_];_]
    [80; 81) 'b': [[&str;_];_]
    [83; 84) 'b': [[&str;_];_]
    [96; 97) 'd': [isize;_]
    [100; 112) '[y, 1, 2, 3]': [isize;_]
    [101; 102) 'y': isize
    [104; 105) '1': isize
    [107; 108) '2': isize
    [110; 111) '3': isize
    [122; 123) 'd': [isize;_]
    [126; 138) '[1, y, 2, 3]': [isize;_]
    [127; 128) '1': isize
    [130; 131) 'y': isize
    [133; 134) '2': isize
    [136; 137) '3': isize
    [148; 149) 'e': [isize;_]
    [152; 155) '[y]': [isize;_]
    [153; 154) 'y': isize
    [165; 166) 'f': [[isize;_];_]
    [169; 175) '[d, d]': [[isize;_];_]
    [170; 171) 'd': [isize;_]
    [173; 174) 'd': [isize;_]
    [185; 186) 'g': [[isize;_];_]
    [189; 195) '[e, e]': [[isize;_];_]
    [190; 191) 'e': [isize;_]
    [193; 194) 'e': [isize;_]
    [206; 207) 'h': [i32;_]
    [210; 216) '[1, 2]': [i32;_]
    [211; 212) '1': i32
    [214; 215) '2': i32
    [226; 227) 'i': [&str;_]
    [230; 240) '["a", "b"]': [&str;_]
    [231; 234) '"a"': &str
    [236; 239) '"b"': &str
    [251; 252) 'b': [[&str;_];_]
    [255; 265) '[a, ["b"]]': [[&str;_];_]
    [256; 257) 'a': [&str;_]
    [259; 264) '["b"]': [&str;_]
    [260; 263) '"b"': &str
    [275; 276) 'x': [u8;_]
    [288; 290) '[]': [u8;_]
    "###
    );
}

#[test]
fn infer_struct_generics() {
    assert_snapshot!(
        infer(r#"
struct A<T> {
    x: T,
}

fn test(a1: A<u32>, i: i32) {
    a1.x;
    let a2 = A { x: i };
    a2.x;
    let a3 = A::<i128> { x: 1 };
    a3.x;
}
"#),
        @r###"
    [36; 38) 'a1': A<u32>
    [48; 49) 'i': i32
    [56; 147) '{     ...3.x; }': ()
    [62; 64) 'a1': A<u32>
    [62; 66) 'a1.x': u32
    [76; 78) 'a2': A<i32>
    [81; 91) 'A { x: i }': A<i32>
    [88; 89) 'i': i32
    [97; 99) 'a2': A<i32>
    [97; 101) 'a2.x': i32
    [111; 113) 'a3': A<i128>
    [116; 134) 'A::<i1...x: 1 }': A<i128>
    [131; 132) '1': i128
    [140; 142) 'a3': A<i128>
    [140; 144) 'a3.x': i128
    "###
    );
}

#[test]
fn infer_tuple_struct_generics() {
    assert_snapshot!(
        infer(r#"
struct A<T>(T);
enum Option<T> { Some(T), None }
use Option::*;

fn test() {
    A(42);
    A(42u128);
    Some("x");
    Option::Some("x");
    None;
    let x: Option<i64> = None;
}
"#),
        @r###"
    [76; 184) '{     ...one; }': ()
    [82; 83) 'A': A<i32>(T) -> A<T>
    [82; 87) 'A(42)': A<i32>
    [84; 86) '42': i32
    [93; 94) 'A': A<u128>(T) -> A<T>
    [93; 102) 'A(42u128)': A<u128>
    [95; 101) '42u128': u128
    [108; 112) 'Some': Some<&str>(T) -> Option<T>
    [108; 117) 'Some("x")': Option<&str>
    [113; 116) '"x"': &str
    [123; 135) 'Option::Some': Some<&str>(T) -> Option<T>
    [123; 140) 'Option...e("x")': Option<&str>
    [136; 139) '"x"': &str
    [146; 150) 'None': Option<{unknown}>
    [160; 161) 'x': Option<i64>
    [177; 181) 'None': Option<i64>
    "###
    );
}

#[test]
fn infer_function_generics() {
    assert_snapshot!(
        infer(r#"
fn id<T>(t: T) -> T { t }

fn test() {
    id(1u32);
    id::<i128>(1);
    let x: u64 = id(1);
}
"#),
        @r###"
    [10; 11) 't': T
    [21; 26) '{ t }': T
    [23; 24) 't': T
    [38; 98) '{     ...(1); }': ()
    [44; 46) 'id': fn id<u32>(T) -> T
    [44; 52) 'id(1u32)': u32
    [47; 51) '1u32': u32
    [58; 68) 'id::<i128>': fn id<i128>(T) -> T
    [58; 71) 'id::<i128>(1)': i128
    [69; 70) '1': i128
    [81; 82) 'x': u64
    [90; 92) 'id': fn id<u64>(T) -> T
    [90; 95) 'id(1)': u64
    [93; 94) '1': u64
    "###
    );
}

#[test]
fn infer_impl_generics() {
    assert_snapshot!(
        infer(r#"
struct A<T1, T2> {
    x: T1,
    y: T2,
}
impl<Y, X> A<X, Y> {
    fn x(self) -> X {
        self.x
    }
    fn y(self) -> Y {
        self.y
    }
    fn z<T>(self, t: T) -> (X, Y, T) {
        (self.x, self.y, t)
    }
}

fn test() -> i128 {
    let a = A { x: 1u64, y: 1i64 };
    a.x();
    a.y();
    a.z(1i128);
    a.z::<u128>(1);
}
"#),
        @r###"
    [74; 78) 'self': A<X, Y>
    [85; 107) '{     ...     }': X
    [95; 99) 'self': A<X, Y>
    [95; 101) 'self.x': X
    [117; 121) 'self': A<X, Y>
    [128; 150) '{     ...     }': Y
    [138; 142) 'self': A<X, Y>
    [138; 144) 'self.y': Y
    [163; 167) 'self': A<X, Y>
    [169; 170) 't': T
    [188; 223) '{     ...     }': (X, Y, T)
    [198; 217) '(self.....y, t)': (X, Y, T)
    [199; 203) 'self': A<X, Y>
    [199; 205) 'self.x': X
    [207; 211) 'self': A<X, Y>
    [207; 213) 'self.y': Y
    [215; 216) 't': T
    [245; 342) '{     ...(1); }': ()
    [255; 256) 'a': A<u64, i64>
    [259; 281) 'A { x:...1i64 }': A<u64, i64>
    [266; 270) '1u64': u64
    [275; 279) '1i64': i64
    [287; 288) 'a': A<u64, i64>
    [287; 292) 'a.x()': u64
    [298; 299) 'a': A<u64, i64>
    [298; 303) 'a.y()': i64
    [309; 310) 'a': A<u64, i64>
    [309; 319) 'a.z(1i128)': (u64, i64, i128)
    [313; 318) '1i128': i128
    [325; 326) 'a': A<u64, i64>
    [325; 339) 'a.z::<u128>(1)': (u64, i64, u128)
    [337; 338) '1': u128
    "###
    );
}

#[test]
fn infer_impl_generics_with_autoderef() {
    assert_snapshot!(
        infer(r#"
enum Option<T> {
    Some(T),
    None,
}
impl<T> Option<T> {
    fn as_ref(&self) -> Option<&T> {}
}
fn test(o: Option<u32>) {
    (&o).as_ref();
    o.as_ref();
}
"#),
        @r###"
    [78; 82) 'self': &Option<T>
    [98; 100) '{}': ()
    [111; 112) 'o': Option<u32>
    [127; 165) '{     ...f(); }': ()
    [133; 146) '(&o).as_ref()': Option<&u32>
    [134; 136) '&o': &Option<u32>
    [135; 136) 'o': Option<u32>
    [152; 153) 'o': Option<u32>
    [152; 162) 'o.as_ref()': Option<&u32>
    "###
    );
}

#[test]
fn infer_generic_chain() {
    assert_snapshot!(
        infer(r#"
struct A<T> {
    x: T,
}
impl<T2> A<T2> {
    fn x(self) -> T2 {
        self.x
    }
}
fn id<T>(t: T) -> T { t }

fn test() -> i128 {
     let x = 1;
     let y = id(x);
     let a = A { x: id(y) };
     let z = id(a.x);
     let b = A { x: z };
     b.x()
}
"#),
        @r###"
    [53; 57) 'self': A<T2>
    [65; 87) '{     ...     }': T2
    [75; 79) 'self': A<T2>
    [75; 81) 'self.x': T2
    [99; 100) 't': T
    [110; 115) '{ t }': T
    [112; 113) 't': T
    [135; 261) '{     ....x() }': i128
    [146; 147) 'x': i128
    [150; 151) '1': i128
    [162; 163) 'y': i128
    [166; 168) 'id': fn id<i128>(T) -> T
    [166; 171) 'id(x)': i128
    [169; 170) 'x': i128
    [182; 183) 'a': A<i128>
    [186; 200) 'A { x: id(y) }': A<i128>
    [193; 195) 'id': fn id<i128>(T) -> T
    [193; 198) 'id(y)': i128
    [196; 197) 'y': i128
    [211; 212) 'z': i128
    [215; 217) 'id': fn id<i128>(T) -> T
    [215; 222) 'id(a.x)': i128
    [218; 219) 'a': A<i128>
    [218; 221) 'a.x': i128
    [233; 234) 'b': A<i128>
    [237; 247) 'A { x: z }': A<i128>
    [244; 245) 'z': i128
    [254; 255) 'b': A<i128>
    [254; 259) 'b.x()': i128
    "###
    );
}

#[test]
fn infer_associated_const() {
    assert_snapshot!(
        infer(r#"
struct Struct;

impl Struct {
    const FOO: u32 = 1;
}

enum Enum {}

impl Enum {
    const BAR: u32 = 2;
}

trait Trait {
    const ID: u32;
}

struct TraitTest;

impl Trait for TraitTest {
    const ID: u32 = 5;
}

fn test() {
    let x = Struct::FOO;
    let y = Enum::BAR;
    let z = TraitTest::ID;
}
"#),
        @r###"
    [52; 53) '1': u32
    [105; 106) '2': u32
    [213; 214) '5': u32
    [229; 307) '{     ...:ID; }': ()
    [239; 240) 'x': u32
    [243; 254) 'Struct::FOO': u32
    [264; 265) 'y': u32
    [268; 277) 'Enum::BAR': u32
    [287; 288) 'z': u32
    [291; 304) 'TraitTest::ID': u32
    "###
    );
}

#[test]
fn infer_type_alias() {
    assert_snapshot!(
        infer(r#"
struct A<X, Y> { x: X, y: Y }
type Foo = A<u32, i128>;
type Bar<T> = A<T, u128>;
type Baz<U, V> = A<V, U>;
fn test(x: Foo, y: Bar<&str>, z: Baz<i8, u8>) {
    x.x;
    x.y;
    y.x;
    y.y;
    z.x;
    z.y;
}
"#),
        @r###"
    [116; 117) 'x': A<u32, i128>
    [124; 125) 'y': A<&str, u128>
    [138; 139) 'z': A<u8, i8>
    [154; 211) '{     ...z.y; }': ()
    [160; 161) 'x': A<u32, i128>
    [160; 163) 'x.x': u32
    [169; 170) 'x': A<u32, i128>
    [169; 172) 'x.y': i128
    [178; 179) 'y': A<&str, u128>
    [178; 181) 'y.x': &str
    [187; 188) 'y': A<&str, u128>
    [187; 190) 'y.y': u128
    [196; 197) 'z': A<u8, i8>
    [196; 199) 'z.x': u8
    [205; 206) 'z': A<u8, i8>
    [205; 208) 'z.y': i8
    "###
    )
}

#[test]
fn recursive_type_alias() {
    assert_snapshot!(
        infer(r#"
struct A<X> {}
type Foo = Foo;
type Bar = A<Bar>;
fn test(x: Foo) {}
"#),
        @r###"
    [59; 60) 'x': {unknown}
    [67; 69) '{}': ()
    "###
    )
}

#[test]
fn infer_type_param() {
    assert_snapshot!(
        infer(r#"
fn id<T>(x: T) -> T {
    x
}

fn clone<T>(x: &T) -> T {
    *x
}

fn test() {
    let y = 10u32;
    id(y);
    let x: bool = clone(z);
    id::<i128>(1);
}
"#),
        @r###"
    [10; 11) 'x': T
    [21; 30) '{     x }': T
    [27; 28) 'x': T
    [44; 45) 'x': &T
    [56; 66) '{     *x }': T
    [62; 64) '*x': T
    [63; 64) 'x': &T
    [78; 158) '{     ...(1); }': ()
    [88; 89) 'y': u32
    [92; 97) '10u32': u32
    [103; 105) 'id': fn id<u32>(T) -> T
    [103; 108) 'id(y)': u32
    [106; 107) 'y': u32
    [118; 119) 'x': bool
    [128; 133) 'clone': fn clone<bool>(&T) -> T
    [128; 136) 'clone(z)': bool
    [134; 135) 'z': &bool
    [142; 152) 'id::<i128>': fn id<i128>(T) -> T
    [142; 155) 'id::<i128>(1)': i128
    [153; 154) '1': i128
    "###
    );
}

#[test]
fn infer_const() {
    assert_snapshot!(
        infer(r#"
struct Foo;
impl Foo { const ASSOC_CONST: u32 = 0; }
const GLOBAL_CONST: u32 = 101;
fn test() {
    const LOCAL_CONST: u32 = 99;
    let x = LOCAL_CONST;
    let z = GLOBAL_CONST;
    let id = Foo::ASSOC_CONST;
}
"#),
        @r###"
    [49; 50) '0': u32
    [80; 83) '101': u32
    [95; 213) '{     ...NST; }': ()
    [138; 139) 'x': u32
    [142; 153) 'LOCAL_CONST': u32
    [163; 164) 'z': u32
    [167; 179) 'GLOBAL_CONST': u32
    [189; 191) 'id': u32
    [194; 210) 'Foo::A..._CONST': u32
    [126; 128) '99': u32
    "###
    );
}

#[test]
fn infer_static() {
    assert_snapshot!(
        infer(r#"
static GLOBAL_STATIC: u32 = 101;
static mut GLOBAL_STATIC_MUT: u32 = 101;
fn test() {
    static LOCAL_STATIC: u32 = 99;
    static mut LOCAL_STATIC_MUT: u32 = 99;
    let x = LOCAL_STATIC;
    let y = LOCAL_STATIC_MUT;
    let z = GLOBAL_STATIC;
    let w = GLOBAL_STATIC_MUT;
}
"#),
        @r###"
    [29; 32) '101': u32
    [70; 73) '101': u32
    [85; 280) '{     ...MUT; }': ()
    [173; 174) 'x': u32
    [177; 189) 'LOCAL_STATIC': u32
    [199; 200) 'y': u32
    [203; 219) 'LOCAL_...IC_MUT': u32
    [229; 230) 'z': u32
    [233; 246) 'GLOBAL_STATIC': u32
    [256; 257) 'w': u32
    [260; 277) 'GLOBAL...IC_MUT': u32
    [118; 120) '99': u32
    [161; 163) '99': u32
    "###
    );
}

#[test]
fn shadowing_primitive() {
    let t = type_at(
        r#"
//- /main.rs
struct i32;
struct Foo;

impl i32 { fn foo(&self) -> Foo { Foo } }

fn main() {
    let x: i32 = i32;
    x.foo()<|>;
}"#,
    );
    assert_eq!(t, "Foo");
}

#[test]
fn not_shadowing_primitive_by_module() {
    let t = type_at(
        r#"
//- /str.rs
fn foo() {}

//- /main.rs
mod str;
fn foo() -> &'static str { "" }

fn main() {
    foo()<|>;
}"#,
    );
    assert_eq!(t, "&str");
}

#[test]
fn not_shadowing_module_by_primitive() {
    let t = type_at(
        r#"
//- /str.rs
fn foo() -> u32 {0}

//- /main.rs
mod str;
fn foo() -> &'static str { "" }

fn main() {
    str::foo()<|>;
}"#,
    );
    assert_eq!(t, "u32");
}

#[test]
fn closure_return() {
    assert_snapshot!(
        infer(r#"
fn foo() -> u32 {
    let x = || -> usize { return 1; };
}
"#),
        @r###"
    [17; 59) '{     ...; }; }': ()
    [27; 28) 'x': || -> usize
    [31; 56) '|| -> ...n 1; }': || -> usize
    [43; 56) '{ return 1; }': !
    [45; 53) 'return 1': !
    [52; 53) '1': usize
    "###
    );
}

#[test]
fn closure_return_unit() {
    assert_snapshot!(
        infer(r#"
fn foo() -> u32 {
    let x = || { return; };
}
"#),
        @r###"
    [17; 48) '{     ...; }; }': ()
    [27; 28) 'x': || -> ()
    [31; 45) '|| { return; }': || -> ()
    [34; 45) '{ return; }': !
    [36; 42) 'return': !
    "###
    );
}

#[test]
fn closure_return_inferred() {
    assert_snapshot!(
        infer(r#"
fn foo() -> u32 {
    let x = || { "test" };
}
"#),
        @r###"
    [17; 47) '{     ..." }; }': ()
    [27; 28) 'x': || -> &str
    [31; 44) '|| { "test" }': || -> &str
    [34; 44) '{ "test" }': &str
    [36; 42) '"test"': &str
    "###
    );
}
