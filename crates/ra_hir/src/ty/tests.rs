use std::fmt::Write;
use std::sync::Arc;

use insta::assert_snapshot_matches;

use ra_db::{salsa::Database, FilePosition, SourceDatabase};
use ra_syntax::{
    algo,
    ast::{self, AstNode},
    SyntaxKind::*,
};
use test_utils::covers;

use crate::{
    expr::BodySourceMap, mock::MockDatabase, ty::display::HirDisplay, ty::InferenceResult,
    SourceAnalyzer,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results using insta. Use cargo-insta to
// update the snapshots.

#[test]
fn infer_await() {
    let (mut db, pos) = MockDatabase::with_position(
        r#"
//- /main.rs

struct IntFuture;

impl Future for IntFuture {
    type Output = u64;
}

fn test() {
    let r = IntFuture;
    let v = r.await;
    v<|>;
}

//- /std.rs
#[prelude_import] use future::*;
mod future {
    trait Future {
        type Output;
    }
}

"#,
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["std"]),
        "std": ("/std.rs", []),
    });
    assert_eq!("u64", type_at_pos(&db, pos));
}

#[test]
fn infer_try() {
    let (mut db, pos) = MockDatabase::with_position(
        r#"
//- /main.rs

fn test() {
    let r: Result<i32, u64> = Result::Ok(1);
    let v = r?;
    v<|>;
}

//- /std.rs

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
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["std"]),
        "std": ("/std.rs", []),
    });
    assert_eq!("i32", type_at_pos(&db, pos));
}

#[test]
fn infer_for_loop() {
    let (mut db, pos) = MockDatabase::with_position(
        r#"
//- /main.rs

use std::collections::Vec;

fn test() {
    let v = Vec::new();
    v.push("foo");
    for x in v {
        x<|>;
    }
}

//- /std.rs

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
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["std"]),
        "std": ("/std.rs", []),
    });
    assert_eq!("&str", type_at_pos(&db, pos));
}

#[test]
fn infer_basics() {
    assert_snapshot_matches!(
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
[42; 121) '{     ...f32; }': ()
[48; 49) 'a': u32
[55; 56) 'b': isize
[62; 63) 'c': !
[69; 70) 'd': &str
[76; 82) '1usize': usize
[88; 94) '1isize': isize
[100; 106) '"test"': &str
[112; 118) '1.0f32': f32"###
    );
}

#[test]
fn infer_let() {
    assert_snapshot_matches!(
        infer(r#"
fn test() {
    let a = 1isize;
    let b: usize = 1;
    let c = b;
}
"#),
        @r###"
[11; 71) '{     ...= b; }': ()
[21; 22) 'a': isize
[25; 31) '1isize': isize
[41; 42) 'b': usize
[52; 53) '1': usize
[63; 64) 'c': usize
[67; 68) 'b': usize"###
    );
}

#[test]
fn infer_paths() {
    assert_snapshot_matches!(
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
[82; 88) 'b::c()': u32"###
    );
}

#[test]
fn infer_struct() {
    assert_snapshot_matches!(
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
[148; 151) 'a.c': C"###
    );
}

#[test]
fn infer_enum() {
    assert_snapshot_matches!(
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
[74; 79) 'E::V2': E"###
    );
}

#[test]
fn infer_refs() {
    assert_snapshot_matches!(
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
[146; 147) 'd': *mut u32"###
    );
}

#[test]
fn infer_literals() {
    assert_snapshot_matches!(
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
[208; 218) 'br#"yolo"#': &[u8]"###
    );
}

#[test]
fn infer_unary_op() {
    assert_snapshot_matches!(
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
[262; 269) '"hello"': &str"###
    );
}

#[test]
fn infer_backwards() {
    assert_snapshot_matches!(
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
[228; 229) 'c': f64"###
    );
}

#[test]
fn infer_self() {
    assert_snapshot_matches!(
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
[187; 194) 'Self {}': S"###
    );
}

#[test]
fn infer_binary_op() {
    assert_snapshot_matches!(
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
[367; 368) '3': usize"###
    );
}

#[test]
fn infer_field_autoderef() {
    assert_snapshot_matches!(
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
[266; 270) 'a2.b': B"###
    );
}

#[test]
fn bug_484() {
    assert_snapshot_matches!(
        infer(r#"
fn test() {
   let x = if true {};
}
"#),
        @r###"
[11; 37) '{    l... {}; }': ()
[20; 21) 'x': ()
[24; 34) 'if true {}': ()
[27; 31) 'true': bool
[32; 34) '{}': ()"###
    );
}

#[test]
fn infer_in_elseif() {
    assert_snapshot_matches!(
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
[73; 107) 'if fal...     }': i32
[76; 81) 'false': bool
[82; 107) '{     ...     }': i32
[92; 95) 'foo': Foo
[92; 101) 'foo.field': i32"###
    )
}

#[test]
fn infer_inherent_method() {
    assert_snapshot_matches!(
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
[193; 194) '1': u64"###
    );
}

#[test]
fn infer_inherent_method_str() {
    assert_snapshot_matches!(
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
[75; 86) '"foo".foo()': i32"###
    );
}

#[test]
fn infer_tuple() {
    assert_snapshot_matches!(
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
[163; 166) '"d"': &str"###
    );
}

#[test]
fn infer_array() {
    assert_snapshot_matches!(
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
    let z: &[u8] = &[1, 2, 3];
}
"#),
        @r###"
[9; 10) 'x': &str
[18; 19) 'y': isize
[28; 324) '{     ... 3]; }': ()
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
[300; 301) 'z': &[u8;_]
[311; 321) '&[1, 2, 3]': &[u8;_]
[312; 321) '[1, 2, 3]': [u8;_]
[313; 314) '1': u8
[316; 317) '2': u8
[319; 320) '3': u8"###
    );
}

#[test]
fn infer_pattern() {
    assert_snapshot_matches!(
        infer(r#"
fn test(x: &i32) {
    let y = x;
    let &z = x;
    let a = z;
    let (c, d) = (1, "hello");

    for (e, f) in some_iter {
        let g = e;
    }

    if let [val] = opt {
        let h = val;
    }

    let lambda = |a: u64, b, c: i32| { a + b; c };

    let ref ref_to_x = x;
    let mut mut_x = x;
    let ref mut mut_ref_to_x = x;
    let k = mut_ref_to_x;
}
"#),
        @r###"
[9; 10) 'x': &i32
[18; 369) '{     ...o_x; }': ()
[28; 29) 'y': &i32
[32; 33) 'x': &i32
[43; 45) '&z': &i32
[44; 45) 'z': i32
[48; 49) 'x': &i32
[59; 60) 'a': i32
[63; 64) 'z': i32
[74; 80) '(c, d)': (i32, &str)
[75; 76) 'c': i32
[78; 79) 'd': &str
[83; 95) '(1, "hello")': (i32, &str)
[84; 85) '1': i32
[87; 94) '"hello"': &str
[102; 152) 'for (e...     }': ()
[106; 112) '(e, f)': ({unknown}, {unknown})
[107; 108) 'e': {unknown}
[110; 111) 'f': {unknown}
[116; 125) 'some_iter': {unknown}
[126; 152) '{     ...     }': ()
[140; 141) 'g': {unknown}
[144; 145) 'e': {unknown}
[158; 205) 'if let...     }': ()
[165; 170) '[val]': {unknown}
[173; 176) 'opt': {unknown}
[177; 205) '{     ...     }': ()
[191; 192) 'h': {unknown}
[195; 198) 'val': {unknown}
[215; 221) 'lambda': {unknown}
[224; 256) '|a: u6...b; c }': {unknown}
[225; 226) 'a': u64
[233; 234) 'b': u64
[236; 237) 'c': i32
[244; 256) '{ a + b; c }': i32
[246; 247) 'a': u64
[246; 251) 'a + b': u64
[250; 251) 'b': u64
[253; 254) 'c': i32
[267; 279) 'ref ref_to_x': &&i32
[282; 283) 'x': &i32
[293; 302) 'mut mut_x': &i32
[305; 306) 'x': &i32
[316; 336) 'ref mu...f_to_x': &mut &i32
[339; 340) 'x': &i32
[350; 351) 'k': &mut &i32
[354; 366) 'mut_ref_to_x': &mut &i32"###
    );
}

#[test]
fn infer_pattern_match_ergonomics() {
    assert_snapshot_matches!(
        infer(r#"
struct A<T>(T);

fn test() {
    let A(n) = &A(1);
    let A(n) = &mut A(1);
}
"#),
    @r###"
[28; 79) '{     ...(1); }': ()
[38; 42) 'A(n)': A<i32>
[40; 41) 'n': &i32
[45; 50) '&A(1)': &A<i32>
[46; 47) 'A': A<i32>(T) -> A<T>
[46; 50) 'A(1)': A<i32>
[48; 49) '1': i32
[60; 64) 'A(n)': A<i32>
[62; 63) 'n': &mut i32
[67; 76) '&mut A(1)': &mut A<i32>
[72; 73) 'A': A<i32>(T) -> A<T>
[72; 76) 'A(1)': A<i32>
[74; 75) '1': i32"###
    );
}

#[test]
fn infer_pattern_match_ergonomics_ref() {
    covers!(match_ergonomics_ref);
    assert_snapshot_matches!(
        infer(r#"
fn test() {
    let v = &(1, &2);
    let (_, &w) = v;
}
"#),
    @r###"
[11; 57) '{     ...= v; }': ()
[21; 22) 'v': &(i32, &i32)
[25; 33) '&(1, &2)': &(i32, &i32)
[26; 33) '(1, &2)': (i32, &i32)
[27; 28) '1': i32
[30; 32) '&2': &i32
[31; 32) '2': i32
[43; 50) '(_, &w)': (i32, &i32)
[44; 45) '_': i32
[47; 49) '&w': &i32
[48; 49) 'w': i32
[53; 54) 'v': &(i32, &i32)"###
    );
}

#[test]
fn infer_adt_pattern() {
    assert_snapshot_matches!(
        infer(r#"
enum E {
    A { x: usize },
    B
}

struct S(u32, E);

fn test() {
    let e = E::A { x: 3 };

    let S(y, z) = foo;
    let E::A { x: new_var } = e;

    match e {
        E::A { x } => x,
        E::B if foo => 1,
        E::B => 10,
    };

    let ref d @ E::A { .. } = e;
    d;
}
"#),
        @r###"
[68; 289) '{     ...  d; }': ()
[78; 79) 'e': E
[82; 95) 'E::A { x: 3 }': E
[92; 93) '3': usize
[106; 113) 'S(y, z)': S
[108; 109) 'y': u32
[111; 112) 'z': E
[116; 119) 'foo': S
[129; 148) 'E::A {..._var }': E
[139; 146) 'new_var': usize
[151; 152) 'e': E
[159; 245) 'match ...     }': usize
[165; 166) 'e': E
[177; 187) 'E::A { x }': E
[184; 185) 'x': usize
[191; 192) 'x': usize
[202; 206) 'E::B': E
[210; 213) 'foo': bool
[217; 218) '1': usize
[228; 232) 'E::B': E
[236; 238) '10': usize
[256; 275) 'ref d ...{ .. }': &E
[264; 275) 'E::A { .. }': E
[278; 279) 'e': E
[285; 286) 'd': &E"###
    );
}

#[test]
fn infer_struct_generics() {
    assert_snapshot_matches!(
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
[140; 144) 'a3.x': i128"###
    );
}

#[test]
fn infer_tuple_struct_generics() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[76; 184) '{     ...one; }': ()
   ⋮[82; 83) 'A': A<i32>(T) -> A<T>
   ⋮[82; 87) 'A(42)': A<i32>
   ⋮[84; 86) '42': i32
   ⋮[93; 94) 'A': A<u128>(T) -> A<T>
   ⋮[93; 102) 'A(42u128)': A<u128>
   ⋮[95; 101) '42u128': u128
   ⋮[108; 112) 'Some': Some<&str>(T) -> Option<T>
   ⋮[108; 117) 'Some("x")': Option<&str>
   ⋮[113; 116) '"x"': &str
   ⋮[123; 135) 'Option::Some': Some<&str>(T) -> Option<T>
   ⋮[123; 140) 'Option...e("x")': Option<&str>
   ⋮[136; 139) '"x"': &str
   ⋮[146; 150) 'None': Option<{unknown}>
   ⋮[160; 161) 'x': Option<i64>
   ⋮[177; 181) 'None': Option<i64>
    "###
    );
}

#[test]
fn infer_generics_in_patterns() {
    assert_snapshot_matches!(
        infer(r#"
struct A<T> {
    x: T,
}

enum Option<T> {
    Some(T),
    None,
}

fn test(a1: A<u32>, o: Option<u64>) {
    let A { x: x2 } = a1;
    let A::<i64> { x: x3 } = A { x: 1 };
    match o {
        Option::Some(t) => t,
        _ => 1,
    };
}
"#),
        @r###"
[79; 81) 'a1': A<u32>
[91; 92) 'o': Option<u64>
[107; 244) '{     ...  }; }': ()
[117; 128) 'A { x: x2 }': A<u32>
[124; 126) 'x2': u32
[131; 133) 'a1': A<u32>
[143; 161) 'A::<i6...: x3 }': A<i64>
[157; 159) 'x3': i64
[164; 174) 'A { x: 1 }': A<i64>
[171; 172) '1': i64
[180; 241) 'match ...     }': u64
[186; 187) 'o': Option<u64>
[198; 213) 'Option::Some(t)': Option<u64>
[211; 212) 't': u64
[217; 218) 't': u64
[228; 229) '_': Option<u64>
[233; 234) '1': u64"###
    );
}

#[test]
fn infer_function_generics() {
    assert_snapshot_matches!(
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
[93; 94) '1': u64"###
    );
}

#[test]
fn infer_impl_generics() {
    assert_snapshot_matches!(
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
[337; 338) '1': u128"###
    );
}

#[test]
fn infer_impl_generics_with_autoderef() {
    assert_snapshot_matches!(
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
[152; 162) 'o.as_ref()': Option<&u32>"###
    );
}

#[test]
fn infer_generic_chain() {
    assert_snapshot_matches!(
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
[254; 259) 'b.x()': i128"###
    );
}

#[test]
fn infer_associated_const() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[52; 53) '1': u32
   ⋮[105; 106) '2': u32
   ⋮[213; 214) '5': u32
   ⋮[229; 307) '{     ...:ID; }': ()
   ⋮[239; 240) 'x': u32
   ⋮[243; 254) 'Struct::FOO': u32
   ⋮[264; 265) 'y': u32
   ⋮[268; 277) 'Enum::BAR': u32
   ⋮[287; 288) 'z': {unknown}
   ⋮[291; 304) 'TraitTest::ID': {unknown}
    "###
    );
}

#[test]
fn infer_associated_method_struct() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[49; 75) '{     ...     }': A
   ⋮[59; 69) 'A { x: 0 }': A
   ⋮[66; 67) '0': u32
   ⋮[88; 122) '{     ...a.x; }': ()
   ⋮[98; 99) 'a': A
   ⋮[102; 108) 'A::new': fn new() -> A
   ⋮[102; 110) 'A::new()': A
   ⋮[116; 117) 'a': A
   ⋮[116; 119) 'a.x': u32
    "###
    );
}

#[test]
fn infer_associated_method_enum() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[47; 67) '{     ...     }': A
   ⋮[57; 61) 'A::B': A
   ⋮[88; 108) '{     ...     }': A
   ⋮[98; 102) 'A::C': A
   ⋮[121; 178) '{     ...  c; }': ()
   ⋮[131; 132) 'a': A
   ⋮[135; 139) 'A::b': fn b() -> A
   ⋮[135; 141) 'A::b()': A
   ⋮[147; 148) 'a': A
   ⋮[158; 159) 'c': A
   ⋮[162; 166) 'A::c': fn c() -> A
   ⋮[162; 168) 'A::c()': A
   ⋮[174; 175) 'c': A
    "###
    );
}

#[test]
fn infer_associated_method_with_modules() {
    assert_snapshot_matches!(
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
[309; 322) 'c::C::thing()': C"###
    );
}

#[test]
fn infer_associated_method_generics() {
    assert_snapshot_matches!(
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
[147; 151) '0u32': u32"###
    );
}

#[test]
fn infer_associated_method_generics_with_default_param() {
    assert_snapshot_matches!(
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
[80; 104) '{     ...     }': !
[90; 98) 'loop { }': !
[95; 98) '{ }': ()
[118; 146) '{     ...e(); }': ()
[128; 129) 'a': Gen<u32>
[132; 141) 'Gen::make': fn make<u32>() -> Gen<T>
[132; 143) 'Gen::make()': Gen<u32>"###
    );
}

#[test]
fn infer_associated_method_generics_without_args() {
    assert_snapshot_matches!(
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
[76; 100) '{     ...     }': !
[86; 94) 'loop { }': !
[91; 94) '{ }': ()
[114; 149) '{     ...e(); }': ()
[124; 125) 'a': Gen<u32>
[128; 144) 'Gen::<...::make': fn make<u32>() -> Gen<T>
[128; 146) 'Gen::<...make()': Gen<u32>"###
    );
}

#[test]
fn infer_associated_method_generics_2_type_params_without_args() {
    assert_snapshot_matches!(
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
[102; 126) '{     ...     }': !
[112; 120) 'loop { }': !
[117; 120) '{ }': ()
[140; 180) '{     ...e(); }': ()
[150; 151) 'a': Gen<u32, u64>
[154; 175) 'Gen::<...::make': fn make<u64>() -> Gen<u32, T>
[154; 177) 'Gen::<...make()': Gen<u32, u64>"###
    );
}

#[test]
fn infer_type_alias() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[116; 117) 'x': A<u32, i128>
   ⋮[124; 125) 'y': A<&str, u128>
   ⋮[138; 139) 'z': A<u8, i8>
   ⋮[154; 211) '{     ...z.y; }': ()
   ⋮[160; 161) 'x': A<u32, i128>
   ⋮[160; 163) 'x.x': u32
   ⋮[169; 170) 'x': A<u32, i128>
   ⋮[169; 172) 'x.y': i128
   ⋮[178; 179) 'y': A<&str, u128>
   ⋮[178; 181) 'y.x': &str
   ⋮[187; 188) 'y': A<&str, u128>
   ⋮[187; 190) 'y.y': u128
   ⋮[196; 197) 'z': A<u8, i8>
   ⋮[196; 199) 'z.x': u8
   ⋮[205; 206) 'z': A<u8, i8>
   ⋮[205; 208) 'z.y': i8
    "###
    )
}

#[test]
#[should_panic] // we currently can't handle this
fn recursive_type_alias() {
    assert_snapshot_matches!(
        infer(r#"
struct A<X> {}
type Foo = Foo;
type Bar = A<Bar>;
fn test(x: Foo) {}
"#),
        @""
    )
}

#[test]
fn no_panic_on_field_of_enum() {
    assert_snapshot_matches!(
        infer(r#"
enum X {}

fn test(x: X) {
    x.some_field;
}
"#),
        @r###"
[20; 21) 'x': X
[26; 47) '{     ...eld; }': ()
[32; 33) 'x': X
[32; 44) 'x.some_field': {unknown}"###
    );
}

#[test]
fn bug_585() {
    assert_snapshot_matches!(
        infer(r#"
fn test() {
    X {};
    match x {
        A::B {} => (),
        A::Y() => (),
    }
}
"#),
        @r###"
[11; 89) '{     ...   } }': ()
[17; 21) 'X {}': {unknown}
[27; 87) 'match ...     }': ()
[33; 34) 'x': {unknown}
[45; 52) 'A::B {}': {unknown}
[56; 58) '()': ()
[68; 74) 'A::Y()': {unknown}
[78; 80) '()': ()"###
    );
}

#[test]
fn bug_651() {
    assert_snapshot_matches!(
        infer(r#"
fn quux() {
    let y = 92;
    1 + y;
}
"#),
        @r###"
[11; 41) '{     ...+ y; }': ()
[21; 22) 'y': i32
[25; 27) '92': i32
[33; 34) '1': i32
[33; 38) '1 + y': i32
[37; 38) 'y': i32"###
    );
}

#[test]
fn recursive_vars() {
    covers!(type_var_cycles_resolve_completely);
    covers!(type_var_cycles_resolve_as_possible);
    assert_snapshot_matches!(
        infer(r#"
fn test() {
    let y = unknown;
    [y, &y];
}
"#),
        @r###"
[11; 48) '{     ...&y]; }': ()
[21; 22) 'y': &{unknown}
[25; 32) 'unknown': &{unknown}
[38; 45) '[y, &y]': [&&{unknown};_]
[39; 40) 'y': &{unknown}
[42; 44) '&y': &&{unknown}
[43; 44) 'y': &{unknown}"###
    );
}

#[test]
fn recursive_vars_2() {
    covers!(type_var_cycles_resolve_completely);
    covers!(type_var_cycles_resolve_as_possible);
    assert_snapshot_matches!(
        infer(r#"
fn test() {
    let x = unknown;
    let y = unknown;
    [(x, y), (&y, &x)];
}
"#),
        @r###"
[11; 80) '{     ...x)]; }': ()
[21; 22) 'x': &&{unknown}
[25; 32) 'unknown': &&{unknown}
[42; 43) 'y': &&{unknown}
[46; 53) 'unknown': &&{unknown}
[59; 77) '[(x, y..., &x)]': [(&&{unknown}, &&{unknown});_]
[60; 66) '(x, y)': (&&{unknown}, &&{unknown})
[61; 62) 'x': &&{unknown}
[64; 65) 'y': &&{unknown}
[68; 76) '(&y, &x)': (&&&{unknown}, &&&{unknown})
[69; 71) '&y': &&&{unknown}
[70; 71) 'y': &&{unknown}
[73; 75) '&x': &&&{unknown}
[74; 75) 'x': &&{unknown}"###
    );
}

#[test]
fn infer_type_param() {
    assert_snapshot_matches!(
        infer(r#"
fn id<T>(x: T) -> T {
    x
}

fn clone<T>(x: &T) -> T {
    x
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
[56; 65) '{     x }': &T
[62; 63) 'x': &T
[77; 157) '{     ...(1); }': ()
[87; 88) 'y': u32
[91; 96) '10u32': u32
[102; 104) 'id': fn id<u32>(T) -> T
[102; 107) 'id(y)': u32
[105; 106) 'y': u32
[117; 118) 'x': bool
[127; 132) 'clone': fn clone<bool>(&T) -> T
[127; 135) 'clone(z)': bool
[133; 134) 'z': &bool
[141; 151) 'id::<i128>': fn id<i128>(T) -> T
[141; 154) 'id::<i128>(1)': i128
[152; 153) '1': i128"###
    );
}

#[test]
fn infer_std_crash_1() {
    // caused stack overflow, taken from std
    assert_snapshot_matches!(
        infer(r#"
enum Maybe<T> {
    Real(T),
    Fake,
}

fn write() {
    match something_unknown {
        Maybe::Real(ref mut something) => (),
    }
}
"#),
        @r###"
[54; 139) '{     ...   } }': ()
[60; 137) 'match ...     }': ()
[66; 83) 'someth...nknown': Maybe<{unknown}>
[94; 124) 'Maybe:...thing)': Maybe<{unknown}>
[106; 123) 'ref mu...ething': &mut {unknown}
[128; 130) '()': ()"###
    );
}

#[test]
fn infer_std_crash_2() {
    covers!(type_var_resolves_to_int_var);
    // caused "equating two type variables, ...", taken from std
    assert_snapshot_matches!(
        infer(r#"
fn test_line_buffer() {
    &[0, b'\n', 1, b'\n'];
}
"#),
        @r###"
[23; 53) '{     ...n']; }': ()
[29; 50) '&[0, b...b'\n']': &[u8;_]
[30; 50) '[0, b'...b'\n']': [u8;_]
[31; 32) '0': u8
[34; 39) 'b'\n'': u8
[41; 42) '1': u8
[44; 49) 'b'\n'': u8"###
    );
}

#[test]
fn infer_std_crash_3() {
    // taken from rustc
    assert_snapshot_matches!(
        infer(r#"
pub fn compute() {
    match nope!() {
        SizeSkeleton::Pointer { non_zero: true, tail } => {}
    }
}
"#),
        @r###"
   ⋮
   ⋮[18; 108) '{     ...   } }': ()
   ⋮[24; 106) 'match ...     }': ()
   ⋮[30; 37) 'nope!()': {unknown}
   ⋮[48; 94) 'SizeSk...tail }': {unknown}
   ⋮[82; 86) 'true': {unknown}
   ⋮[88; 92) 'tail': {unknown}
   ⋮[98; 100) '{}': ()
    "###
    );
}

#[test]
fn infer_std_crash_4() {
    // taken from rustc
    assert_snapshot_matches!(
        infer(r#"
pub fn primitive_type() {
    match *self {
        BorrowedRef { type_: Primitive(p), ..} => {},
    }
}
"#),
        @r###"
   ⋮
   ⋮[25; 106) '{     ...   } }': ()
   ⋮[31; 104) 'match ...     }': ()
   ⋮[37; 42) '*self': {unknown}
   ⋮[38; 42) 'self': {unknown}
   ⋮[53; 91) 'Borrow...), ..}': {unknown}
   ⋮[74; 86) 'Primitive(p)': {unknown}
   ⋮[84; 85) 'p': {unknown}
   ⋮[95; 97) '{}': ()
    "###
    );
}

#[test]
fn infer_std_crash_5() {
    // taken from rustc
    assert_snapshot_matches!(
        infer(r#"
fn extra_compiler_flags() {
    for content in doesnt_matter {
        let name = if doesnt_matter {
            first
        } else {
            &content
        };

        let content = if ICE_REPORT_COMPILER_FLAGS_STRIP_VALUE.contains(&name) {
            name
        } else {
            content
        };
    }
}
"#),
        @r###"
[27; 323) '{     ...   } }': ()
[33; 321) 'for co...     }': ()
[37; 44) 'content': &{unknown}
[48; 61) 'doesnt_matter': {unknown}
[62; 321) '{     ...     }': ()
[76; 80) 'name': &&{unknown}
[83; 167) 'if doe...     }': &&{unknown}
[86; 99) 'doesnt_matter': bool
[100; 129) '{     ...     }': &&{unknown}
[114; 119) 'first': &&{unknown}
[135; 167) '{     ...     }': &&{unknown}
[149; 157) '&content': &&{unknown}
[150; 157) 'content': &{unknown}
[182; 189) 'content': &&{unknown}
[192; 314) 'if ICE...     }': &&{unknown}
[195; 232) 'ICE_RE..._VALUE': {unknown}
[195; 248) 'ICE_RE...&name)': bool
[242; 247) '&name': &&&{unknown}
[243; 247) 'name': &&{unknown}
[249; 277) '{     ...     }': &&{unknown}
[263; 267) 'name': &&{unknown}
[283; 314) '{     ...     }': &{unknown}
[297; 304) 'content': &{unknown}"###
    );
}

#[test]
fn infer_nested_generics_crash() {
    // another crash found typechecking rustc
    assert_snapshot_matches!(
        infer(r#"
struct Canonical<V> {
    value: V,
}
struct QueryResponse<V> {
    value: V,
}
fn test<R>(query_response: Canonical<QueryResponse<R>>) {
    &query_response.value;
}
"#),
        @r###"
[92; 106) 'query_response': Canonical<QueryResponse<R>>
[137; 167) '{     ...lue; }': ()
[143; 164) '&query....value': &QueryResponse<R>
[144; 158) 'query_response': Canonical<QueryResponse<R>>
[144; 164) 'query_....value': QueryResponse<R>"###
    );
}

#[test]
fn bug_1030() {
    assert_snapshot_matches!(infer(r#"
struct HashSet<T, H>;
struct FxHasher;
type FxHashSet<T> = HashSet<T, FxHasher>;

impl<T, H> HashSet<T, H> {
    fn default() -> HashSet<T, H> {}
}

pub fn main_loop() {
    FxHashSet::default();
}
"#),
    @r###"
[144; 146) '{}': ()
[169; 198) '{     ...t(); }': ()
[175; 193) 'FxHash...efault': fn default<{unknown}, FxHasher>() -> HashSet<T, H>
[175; 195) 'FxHash...ault()': HashSet<{unknown}, FxHasher>"###
    );
}

#[test]
fn cross_crate_associated_method_call() {
    let (mut db, pos) = MockDatabase::with_position(
        r#"
//- /main.rs
fn test() {
    let x = other_crate::foo::S::thing();
    x<|>;
}

//- /lib.rs
mod foo {
    struct S;
    impl S {
        fn thing() -> i128 {}
    }
}
"#,
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["other_crate"]),
        "other_crate": ("/lib.rs", []),
    });
    assert_eq!("i128", type_at_pos(&db, pos));
}

#[test]
fn infer_const() {
    assert_snapshot_matches!(
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
[138; 139) 'x': {unknown}
[142; 153) 'LOCAL_CONST': {unknown}
[163; 164) 'z': u32
[167; 179) 'GLOBAL_CONST': u32
[189; 191) 'id': u32
[194; 210) 'Foo::A..._CONST': u32
[126; 128) '99': u32"###
    );
}

#[test]
fn infer_static() {
    assert_snapshot_matches!(
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
[173; 174) 'x': {unknown}
[177; 189) 'LOCAL_STATIC': {unknown}
[199; 200) 'y': {unknown}
[203; 219) 'LOCAL_...IC_MUT': {unknown}
[229; 230) 'z': u32
[233; 246) 'GLOBAL_STATIC': u32
[256; 257) 'w': u32
[260; 277) 'GLOBAL...IC_MUT': u32
[118; 120) '99': u32
[161; 163) '99': u32"###
    );
}

#[test]
fn infer_trait_method_simple() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot_matches!(
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
[203; 214) 'S2.method()': i128"###
    );
}

#[test]
fn infer_trait_method_scoped() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot_matches!(
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
[426; 436) 'S.method()': i128"###
    );
}

#[test]
fn infer_trait_method_generic_1() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot_matches!(
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
[98; 108) 'S.method()': u32"###
    );
}

#[test]
fn infer_trait_method_generic_more_params() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot_matches!(
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
[324; 336) 'S2.method2()': ({unknown}, i16, i8)"###
    );
}

#[test]
fn infer_trait_method_generic_2() {
    // the trait implementation is intentionally incomplete -- it shouldn't matter
    assert_snapshot_matches!(
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
[110; 114) '1u32': u32"###
    );
}

#[test]
fn infer_trait_assoc_method() {
    assert_snapshot_matches!(
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
[105; 121) 'Defaul...efault': {unknown}
[105; 123) 'Defaul...ault()': S
[133; 135) 's2': {unknown}
[138; 148) 'S::default': {unknown}
[138; 150) 'S::default()': {unknown}
[160; 162) 's3': {unknown}
[165; 188) '<S as ...efault': {unknown}
[165; 190) '<S as ...ault()': {unknown}"###
    );
}

#[test]
fn infer_from_bound_1() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[86; 87) 't': T
   ⋮[92; 94) '{}': ()
   ⋮[105; 144) '{     ...(s); }': ()
   ⋮[115; 116) 's': S<u32>
   ⋮[119; 120) 'S': S<u32>(T) -> S<T>
   ⋮[119; 129) 'S(unknown)': S<u32>
   ⋮[121; 128) 'unknown': u32
   ⋮[135; 138) 'foo': fn foo<S<u32>>(T) -> ()
   ⋮[135; 141) 'foo(s)': ()
   ⋮[139; 140) 's': S<u32>
    "###
    );
}

#[test]
fn infer_from_bound_2() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[87; 88) 't': T
   ⋮[98; 100) '{}': ()
   ⋮[111; 163) '{     ...(s); }': ()
   ⋮[121; 122) 's': S<u32>
   ⋮[125; 126) 'S': S<u32>(T) -> S<T>
   ⋮[125; 135) 'S(unknown)': S<u32>
   ⋮[127; 134) 'unknown': u32
   ⋮[145; 146) 'x': u32
   ⋮[154; 157) 'foo': fn foo<u32, S<u32>>(T) -> U
   ⋮[154; 160) 'foo(s)': u32
   ⋮[158; 159) 's': S<u32>
    "###
    );
}

#[test]
fn infer_call_trait_method_on_generic_param_1() {
    assert_snapshot_matches!(
        infer(r#"
trait Trait {
    fn method() -> u32;
}
fn test<T: Trait>(t: T) {
    t.method();
}
"#),
        @r###"
[59; 60) 't': T
[65; 84) '{     ...d(); }': ()
[71; 72) 't': T
[71; 81) 't.method()': {unknown}"###
    );
}

#[test]
fn infer_call_trait_method_on_generic_param_2() {
    assert_snapshot_matches!(
        infer(r#"
trait Trait<T> {
    fn method() -> T;
}
fn test<U, T: Trait<U>>(t: T) {
    t.method();
}
"#),
        @r###"
[66; 67) 't': T
[72; 91) '{     ...d(); }': ()
[78; 79) 't': T
[78; 88) 't.method()': {unknown}"###
    );
}

#[test]
fn infer_with_multiple_trait_impls() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[29; 33) 'self': Self
   ⋮[111; 202) '{     ...(S); }': ()
   ⋮[121; 122) 'x': u32
   ⋮[130; 131) 'S': S
   ⋮[130; 138) 'S.into()': u32
   ⋮[148; 149) 'y': u64
   ⋮[157; 158) 'S': S
   ⋮[157; 165) 'S.into()': u64
   ⋮[175; 176) 'z': {unknown}
   ⋮[179; 196) 'Into::...::into': {unknown}
   ⋮[179; 199) 'Into::...nto(S)': {unknown}
   ⋮[197; 198) 'S': S
    "###
    );
}

#[test]
fn infer_project_associated_type() {
    assert_snapshot_matches!(
        infer(r#"
trait Iterable {
   type Item;
}
struct S;
impl Iterable for S { type Item = u32; }
fn test<T: Iterable>() {
    let x: <S as Iterable>::Item = 1;
    let y: T::Item = no_matter;
}
"#),
        @r###"
[108; 181) '{     ...ter; }': ()
[118; 119) 'x': i32
[145; 146) '1': i32
[156; 157) 'y': {unknown}
[169; 178) 'no_matter': {unknown}"###
    );
}

#[test]
fn infer_associated_type_bound() {
    assert_snapshot_matches!(
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
[90; 97) 'unknown': {unknown}"###
    );
}

#[test]
fn infer_const_body() {
    assert_snapshot_matches!(
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
[52; 53) 'x': u64"###
    );
}

#[test]
fn tuple_struct_fields() {
    assert_snapshot_matches!(
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
[82; 85) 'a.1': u64"###
    );
}

#[test]
fn tuple_struct_with_fn() {
    assert_snapshot_matches!(
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
[60; 67) '|i| 2*i': fn(u32) -> u64
[61; 62) 'i': i32
[64; 65) '2': i32
[64; 67) '2*i': i32
[66; 67) 'i': i32
[78; 79) 'b': u64
[82; 83) 'a': S
[82; 85) 'a.0': fn(u32) -> u64
[82; 88) 'a.0(4)': u64
[86; 87) '4': u32
[94; 95) 'a': S
[94; 97) 'a.0': fn(u32) -> u64
[94; 100) 'a.0(2)': u64
[98; 99) '2': u32"###
    );
}

#[test]
fn infer_macros_expanded() {
    assert_snapshot_matches!(
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
   ⋮
   ⋮[156; 182) '{     ...,2); }': ()
   ⋮[166; 167) 'x': Foo
    "###
    );
}

#[ignore]
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
    let (mut db, pos) = MockDatabase::with_position(
        r#"
//- /main.rs
struct S;
impl Clone for S {}

fn test() {
    S.clone()<|>;
}

//- /lib.rs
#[prelude_import] use foo::*;

mod foo {
    trait Clone {
        fn clone(&self) -> Self;
    }
}
"#,
    );
    db.set_crate_graph_from_fixture(crate_graph! {
        "main": ("/main.rs", ["other_crate"]),
        "other_crate": ("/lib.rs", []),
    });
    assert_eq!("S", type_at_pos(&db, pos));
}

#[test]
fn method_resolution_where_clause_for_unknown_trait() {
    // The blanket impl shouldn't apply because we can't even resolve UnknownTrait
    let t = type_at(
        r#"
//- /main.rs
trait Trait { fn foo(self) -> u128; }
struct S;
impl<T> Trait for T where T: UnknownTrait {}
fn test() { (&S).foo()<|>; }
"#,
    );
    assert_eq!(t, "{unknown}");
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
    assert_eq!(t, "S1");
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
    assert_eq!(t, "S1");
}

#[test]
fn method_resolution_encountering_fn_type() {
    covers!(trait_resolution_on_fn_type);
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
    assert_eq!(t, "{unknown}");
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
    (*s, s.foo())<|>
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

fn type_at_pos(db: &MockDatabase, pos: FilePosition) -> String {
    let file = db.parse(pos.file_id).ok().unwrap();
    let expr = algo::find_node_at_offset::<ast::Expr>(file.syntax(), pos.offset).unwrap();
    let analyzer = SourceAnalyzer::new(db, pos.file_id, expr.syntax(), Some(pos.offset));
    let ty = analyzer.type_of(db, &expr).unwrap();
    ty.display(db).to_string()
}

fn type_at(content: &str) -> String {
    let (db, file_pos) = MockDatabase::with_position(content);
    type_at_pos(&db, file_pos)
}

fn infer(content: &str) -> String {
    let (db, _, file_id) = MockDatabase::with_single_file(content);
    let source_file = db.parse(file_id).ok().unwrap();

    let mut acc = String::new();
    acc.push_str("\n");

    let mut infer_def = |inference_result: Arc<InferenceResult>,
                         body_source_map: Arc<BodySourceMap>| {
        let mut types = Vec::new();

        for (pat, ty) in inference_result.type_of_pat.iter() {
            let syntax_ptr = match body_source_map.pat_syntax(pat) {
                Some(sp) => sp.either(|it| it.syntax_node_ptr(), |it| it.syntax_node_ptr()),
                None => continue,
            };
            types.push((syntax_ptr, ty));
        }

        for (expr, ty) in inference_result.type_of_expr.iter() {
            let syntax_ptr = match body_source_map.expr_syntax(expr) {
                Some(sp) => sp,
                None => continue,
            };
            types.push((syntax_ptr, ty));
        }

        // sort ranges for consistency
        types.sort_by_key(|(ptr, _)| (ptr.range().start(), ptr.range().end()));
        for (syntax_ptr, ty) in &types {
            let node = syntax_ptr.to_node(source_file.syntax());
            let (range, text) = if let Some(self_param) = ast::SelfParam::cast(node.clone()) {
                (self_param.self_kw_token().text_range(), "self".to_string())
            } else {
                (syntax_ptr.range(), node.text().to_string().replace("\n", " "))
            };
            write!(acc, "{} '{}': {}\n", range, ellipsize(text, 15), ty.display(&db)).unwrap();
        }
    };

    for node in source_file.syntax().descendants() {
        if node.kind() == FN_DEF || node.kind() == CONST_DEF || node.kind() == STATIC_DEF {
            let analyzer = SourceAnalyzer::new(&db, file_id, &node, None);
            infer_def(analyzer.inference_result(), analyzer.body_source_map());
        }
    }

    acc.truncate(acc.trim_end().len());
    acc
}

fn ellipsize(mut text: String, max_len: usize) -> String {
    if text.len() <= max_len {
        return text;
    }
    let ellipsis = "...";
    let e_len = ellipsis.len();
    let mut prefix_len = (max_len - e_len) / 2;
    while !text.is_char_boundary(prefix_len) {
        prefix_len += 1;
    }
    let mut suffix_len = max_len - e_len - prefix_len;
    while !text.is_char_boundary(text.len() - suffix_len) {
        suffix_len += 1;
    }
    text.replace_range(prefix_len..text.len() - suffix_len, ellipsis);
    text
}

#[test]
fn typing_whitespace_inside_a_function_should_not_invalidate_types() {
    let (mut db, pos) = MockDatabase::with_position(
        "
        //- /lib.rs
        fn foo() -> i32 {
            <|>1 + 1
        }
    ",
    );
    {
        let file = db.parse(pos.file_id).ok().unwrap();
        let node = file.syntax().token_at_offset(pos.offset).right_biased().unwrap().parent();
        let events = db.log_executed(|| {
            SourceAnalyzer::new(&db, pos.file_id, &node, None);
        });
        assert!(format!("{:?}", events).contains("infer"))
    }

    let new_text = "
        fn foo() -> i32 {
            1
            +
            1
        }
    "
    .to_string();

    db.query_mut(ra_db::FileTextQuery).set(pos.file_id, Arc::new(new_text));

    {
        let file = db.parse(pos.file_id).ok().unwrap();
        let node = file.syntax().token_at_offset(pos.offset).right_biased().unwrap().parent();
        let events = db.log_executed(|| {
            SourceAnalyzer::new(&db, pos.file_id, &node, None);
        });
        assert!(!format!("{:?}", events).contains("infer"), "{:#?}", events)
    }
}

#[test]
fn no_such_field_diagnostics() {
    let diagnostics = MockDatabase::with_files(
        r"
        //- /lib.rs
        struct S { foo: i32, bar: () }
        impl S {
            fn new() -> S {
                S {
                    foo: 92,
                    baz: 62,
                }
            }
        }
        ",
    )
    .diagnostics();

    assert_snapshot_matches!(diagnostics, @r###"
"baz: 62": no such field
"{\n            foo: 92,\n            baz: 62,\n        }": fill structure fields
"###
    );
}
