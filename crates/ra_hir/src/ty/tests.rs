use std::sync::Arc;
use std::fmt::Write;

use insta::assert_snapshot_matches;

use ra_db::{SourceDatabase, salsa::Database, FilePosition};
use ra_syntax::{algo, ast::{self, AstNode}};
use test_utils::covers;

use crate::{
    source_binder,
    mock::MockDatabase,
};

// These tests compare the inference results for all expressions in a file
// against snapshots of the expected results using insta. Use cargo-insta to
// update the snapshots.

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
}"#),
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
}"#),
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
[11; 201) '{     ...o"#; }': ()
[17; 21) '5i32': i32
[27; 34) '"hello"': &str
[40; 48) 'b"bytes"': &[u8]
[54; 57) ''c'': char
[63; 67) 'b'b'': u8
[73; 77) '3.14': f64
[83; 87) '5000': i32
[93; 98) 'false': bool
[104; 108) 'true': bool
[114; 182) 'r#"   ...    "#': &str
[188; 198) 'br#"yolo"#': &[u8]"###
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
[38; 39) 'a': [&str]
[42; 45) '[x]': [&str]
[43; 44) 'x': &str
[55; 56) 'b': [[&str]]
[59; 65) '[a, a]': [[&str]]
[60; 61) 'a': [&str]
[63; 64) 'a': [&str]
[75; 76) 'c': [[[&str]]]
[79; 85) '[b, b]': [[[&str]]]
[80; 81) 'b': [[&str]]
[83; 84) 'b': [[&str]]
[96; 97) 'd': [isize]
[100; 112) '[y, 1, 2, 3]': [isize]
[101; 102) 'y': isize
[104; 105) '1': isize
[107; 108) '2': isize
[110; 111) '3': isize
[122; 123) 'd': [isize]
[126; 138) '[1, y, 2, 3]': [isize]
[127; 128) '1': isize
[130; 131) 'y': isize
[133; 134) '2': isize
[136; 137) '3': isize
[148; 149) 'e': [isize]
[152; 155) '[y]': [isize]
[153; 154) 'y': isize
[165; 166) 'f': [[isize]]
[169; 175) '[d, d]': [[isize]]
[170; 171) 'd': [isize]
[173; 174) 'd': [isize]
[185; 186) 'g': [[isize]]
[189; 195) '[e, e]': [[isize]]
[190; 191) 'e': [isize]
[193; 194) 'e': [isize]
[206; 207) 'h': [i32]
[210; 216) '[1, 2]': [i32]
[211; 212) '1': i32
[214; 215) '2': i32
[226; 227) 'i': [&str]
[230; 240) '["a", "b"]': [&str]
[231; 234) '"a"': &str
[236; 239) '"b"': &str
[251; 252) 'b': [[&str]]
[255; 265) '[a, ["b"]]': [[&str]]
[256; 257) 'a': [&str]
[259; 264) '["b"]': [&str]
[260; 263) '"b"': &str
[275; 276) 'x': [u8]
[288; 290) '[]': [u8]
[300; 301) 'z': &[u8]
[311; 321) '&[1, 2, 3]': &[u8]
[312; 321) '[1, 2, 3]': [u8]
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
enum Option<T> { Some(T), None };
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
[77; 185) '{     ...one; }': ()
[83; 84) 'A': A<i32>(T) -> A<T>
[83; 88) 'A(42)': A<i32>
[85; 87) '42': i32
[94; 95) 'A': A<u128>(T) -> A<T>
[94; 103) 'A(42u128)': A<u128>
[96; 102) '42u128': u128
[109; 113) 'Some': Some<&str>(T) -> Option<T>
[109; 118) 'Some("x")': Option<&str>
[114; 117) '"x"': &str
[124; 136) 'Option::Some': Some<&str>(T) -> Option<T>
[124; 141) 'Option...e("x")': Option<&str>
[137; 140) '"x"': &str
[147; 151) 'None': Option<{unknown}>
[161; 162) 'x': Option<i64>
[178; 182) 'None': Option<i64>"###
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

enum Enum;

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
[227; 305) '{     ...:ID; }': ()
[237; 238) 'x': u32
[241; 252) 'Struct::FOO': u32
[262; 263) 'y': u32
[266; 275) 'Enum::BAR': u32
[285; 286) 'z': u32
[289; 302) 'TraitTest::ID': u32"###
    );
}

#[test]
fn infer_associated_method_struct() {
    assert_snapshot_matches!(
        infer(r#"
struct A { x: u32 };

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
[50; 76) '{     ...     }': A
[60; 70) 'A { x: 0 }': A
[67; 68) '0': u32
[89; 123) '{     ...a.x; }': ()
[99; 100) 'a': A
[103; 109) 'A::new': fn new() -> A
[103; 111) 'A::new()': A
[117; 118) 'a': A
[117; 120) 'a.x': u32"###
    );
}

#[test]
fn infer_associated_method_enum() {
    assert_snapshot_matches!(
        infer(r#"
enum A { B, C };

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
[48; 68) '{     ...     }': A
[58; 62) 'A::B': A
[89; 109) '{     ...     }': A
[99; 103) 'A::C': A
[122; 179) '{     ...  c; }': ()
[132; 133) 'a': A
[136; 140) 'A::b': fn b() -> A
[136; 142) 'A::b()': A
[148; 149) 'a': A
[159; 160) 'c': A
[163; 167) 'A::c': fn c() -> A
[163; 169) 'A::c()': A
[175; 176) 'c': A"###
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
fn infer_type_alias() {
    assert_snapshot_matches!(
        infer(r#"
struct A<X, Y> { x: X, y: Y };
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
[117; 118) 'x': A<u32, i128>
[125; 126) 'y': A<&str, u128>
[139; 140) 'z': A<u8, i8>
[155; 212) '{     ...z.y; }': ()
[161; 162) 'x': A<u32, i128>
[161; 164) 'x.x': u32
[170; 171) 'x': A<u32, i128>
[170; 173) 'x.y': i128
[179; 180) 'y': A<&str, u128>
[179; 182) 'y.x': &str
[188; 189) 'y': A<&str, u128>
[188; 191) 'y.y': u128
[197; 198) 'z': A<u8, i8>
[197; 200) 'z.x': u8
[206; 207) 'z': A<u8, i8>
[206; 209) 'z.y': i8"###
    )
}

#[test]
#[should_panic] // we currently can't handle this
fn recursive_type_alias() {
    assert_snapshot_matches!(
        infer(r#"
struct A<X> {};
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
[38; 45) '[y, &y]': [&&{unknown}]
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
[59; 77) '[(x, y..., &x)]': [(&&{unknown}, &&{unknown})]
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
[29; 50) '&[0, b...b'\n']': &[u8]
[30; 50) '[0, b'...b'\n']': [u8]
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
    match _ {
        SizeSkeleton::Pointer { non_zero: true, tail } => {}
    }
}
"#),
        @r###"
[18; 102) '{     ...   } }': ()
[24; 100) 'match ...     }': ()
[42; 88) 'SizeSk...tail }': {unknown}
[76; 80) 'true': {unknown}
[82; 86) 'tail': {unknown}
[92; 94) '{}': ()"###
    );
}

#[test]
fn infer_std_crash_4() {
    // taken from rustc
    assert_snapshot_matches!(
        infer(r#"
pub fn primitive_type() {
    match *self {
        BorrowedRef { type_: box Primitive(p), ..} => {},
    }
}
"#),
        @r###"
[25; 110) '{     ...   } }': ()
[31; 108) 'match ...     }': ()
[37; 42) '*self': {unknown}
[38; 42) 'self': {unknown}
[53; 95) 'Borrow...), ..}': {unknown}
[74; 77) 'box': {unknown}
[78; 87) 'Primitive': {unknown}
[88; 89) 'p': {unknown}
[99; 101) '{}': ()"###
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
[95; 213) '{     ...NST; }': ()
[138; 139) 'x': {unknown}
[142; 153) 'LOCAL_CONST': {unknown}
[163; 164) 'z': u32
[167; 179) 'GLOBAL_CONST': u32
[189; 191) 'id': u32
[194; 210) 'Foo::A..._CONST': u32"###
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
[85; 280) '{     ...MUT; }': ()
[173; 174) 'x': {unknown}
[177; 189) 'LOCAL_STATIC': {unknown}
[199; 200) 'y': {unknown}
[203; 219) 'LOCAL_...IC_MUT': {unknown}
[229; 230) 'z': u32
[233; 246) 'GLOBAL_STATIC': u32
[256; 257) 'w': u32
[260; 277) 'GLOBAL...IC_MUT': u32"###
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
[31; 35) 'self': &{unknown}
[110; 114) 'self': &{unknown}
[170; 228) '{     ...i128 }': ()
[176; 178) 'S1': S1
[176; 187) 'S1.method()': {unknown}
[203; 205) 'S2': S2
[203; 214) 'S2.method()': {unknown}"###
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
[63; 67) 'self': &{unknown}
[169; 173) 'self': &{unknown}
[300; 337) '{     ...     }': ()
[310; 311) 'S': S
[310; 320) 'S.method()': {unknown}
[416; 454) '{     ...     }': ()
[426; 427) 'S': S
[426; 436) 'S.method()': {unknown}"###
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
[33; 37) 'self': &{unknown}
[92; 111) '{     ...d(); }': ()
[98; 99) 'S': S
[98; 108) 'S.method()': {unknown}"###
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
[33; 37) 'self': &{unknown}
[102; 127) '{     ...d(); }': ()
[108; 109) 'S': S<u32>(T) -> S<T>
[108; 115) 'S(1u32)': S<u32>
[108; 124) 'S(1u32...thod()': {unknown}
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
[86; 87) 't': T
[92; 94) '{}': ()
[105; 144) '{     ...(s); }': ()
[115; 116) 's': S<{unknown}>
[119; 120) 'S': S<{unknown}>(T) -> S<T>
[119; 129) 'S(unknown)': S<{unknown}>
[121; 128) 'unknown': {unknown}
[135; 138) 'foo': fn foo<S<{unknown}>>(T) -> ()
[135; 141) 'foo(s)': ()
[139; 140) 's': S<{unknown}>"###
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
[87; 88) 't': T
[98; 100) '{}': ()
[111; 163) '{     ...(s); }': ()
[121; 122) 's': S<{unknown}>
[125; 126) 'S': S<{unknown}>(T) -> S<T>
[125; 135) 'S(unknown)': S<{unknown}>
[127; 134) 'unknown': {unknown}
[145; 146) 'x': u32
[154; 157) 'foo': fn foo<u32, S<{unknown}>>(T) -> U
[154; 160) 'foo(s)': u32
[158; 159) 's': S<{unknown}>"###
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
impl Into<u32> for S;
impl Into<u64> for S;
fn test() {
    let x: u32 = S.into();
    let y: u64 = S.into();
    let z = Into::<u64>::into(S);
}
"#),
        @r###"
[29; 33) 'self': {unknown}
[107; 198) '{     ...(S); }': ()
[117; 118) 'x': u32
[126; 127) 'S': S
[126; 134) 'S.into()': u32
[144; 145) 'y': u64
[153; 154) 'S': S
[153; 161) 'S.into()': u64
[171; 172) 'z': {unknown}
[175; 192) 'Into::...::into': {unknown}
[175; 195) 'Into::...nto(S)': {unknown}
[193; 194) 'S': S"###
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

fn type_at_pos(db: &MockDatabase, pos: FilePosition) -> String {
    let func = source_binder::function_from_position(db, pos).unwrap();
    let body_source_map = func.body_source_map(db);
    let inference_result = func.infer(db);
    let (_, syntax) = func.source(db);
    let node = algo::find_node_at_offset::<ast::Expr>(syntax.syntax(), pos.offset).unwrap();
    let expr = body_source_map.node_expr(node).unwrap();
    let ty = &inference_result[expr];
    ty.to_string()
}

fn infer(content: &str) -> String {
    let (db, _, file_id) = MockDatabase::with_single_file(content);
    let source_file = db.parse(file_id);
    let mut acc = String::new();
    acc.push_str("\n");
    for fn_def in source_file.syntax().descendants().filter_map(ast::FnDef::cast) {
        let func = source_binder::function_from_source(&db, file_id, fn_def).unwrap();
        let inference_result = func.infer(&db);
        let body_source_map = func.body_source_map(&db);
        let mut types = Vec::new();
        for (pat, ty) in inference_result.type_of_pat.iter() {
            let syntax_ptr = match body_source_map.pat_syntax(pat) {
                Some(sp) => sp,
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
            let node = syntax_ptr.to_node(&source_file);
            write!(
                acc,
                "{} '{}': {}\n",
                syntax_ptr.range(),
                ellipsize(node.text().to_string().replace("\n", " "), 15),
                ty
            )
            .unwrap();
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
    let func = source_binder::function_from_position(&db, pos).unwrap();
    {
        let events = db.log_executed(|| {
            func.infer(&db);
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
        let events = db.log_executed(|| {
            func.infer(&db);
        });
        assert!(!format!("{:?}", events).contains("infer"), "{:#?}", events)
    }
}
