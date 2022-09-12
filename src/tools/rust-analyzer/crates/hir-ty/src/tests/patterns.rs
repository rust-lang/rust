use expect_test::expect;

use super::{check, check_infer, check_infer_with_mismatches, check_types};

#[test]
fn infer_pattern() {
    check_infer(
        r#"
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

            if let x @ true = &true {}

            let lambda = |a: u64, b, c: i32| { a + b; c };

            let ref ref_to_x = x;
            let mut mut_x = x;
            let ref mut mut_ref_to_x = x;
            let k = mut_ref_to_x;
        }
        "#,
        expect![[r#"
            8..9 'x': &i32
            17..400 '{     ...o_x; }': ()
            27..28 'y': &i32
            31..32 'x': &i32
            42..44 '&z': &i32
            43..44 'z': i32
            47..48 'x': &i32
            58..59 'a': i32
            62..63 'z': i32
            73..79 '(c, d)': (i32, &str)
            74..75 'c': i32
            77..78 'd': &str
            82..94 '(1, "hello")': (i32, &str)
            83..84 '1': i32
            86..93 '"hello"': &str
            101..151 'for (e...     }': ()
            105..111 '(e, f)': ({unknown}, {unknown})
            106..107 'e': {unknown}
            109..110 'f': {unknown}
            115..124 'some_iter': {unknown}
            125..151 '{     ...     }': ()
            139..140 'g': {unknown}
            143..144 'e': {unknown}
            157..204 'if let...     }': ()
            160..175 'let [val] = opt': bool
            164..169 '[val]': [{unknown}]
            165..168 'val': {unknown}
            172..175 'opt': [{unknown}]
            176..204 '{     ...     }': ()
            190..191 'h': {unknown}
            194..197 'val': {unknown}
            210..236 'if let...rue {}': ()
            213..233 'let x ... &true': bool
            217..225 'x @ true': &bool
            221..225 'true': bool
            221..225 'true': bool
            228..233 '&true': &bool
            229..233 'true': bool
            234..236 '{}': ()
            246..252 'lambda': |u64, u64, i32| -> i32
            255..287 '|a: u6...b; c }': |u64, u64, i32| -> i32
            256..257 'a': u64
            264..265 'b': u64
            267..268 'c': i32
            275..287 '{ a + b; c }': i32
            277..278 'a': u64
            277..282 'a + b': u64
            281..282 'b': u64
            284..285 'c': i32
            298..310 'ref ref_to_x': &&i32
            313..314 'x': &i32
            324..333 'mut mut_x': &i32
            336..337 'x': &i32
            347..367 'ref mu...f_to_x': &mut &i32
            370..371 'x': &i32
            381..382 'k': &mut &i32
            385..397 'mut_ref_to_x': &mut &i32
        "#]],
    );
}

#[test]
fn infer_literal_pattern() {
    check_infer_with_mismatches(
        r#"
        fn any<T>() -> T { loop {} }
        fn test(x: &i32) {
            if let "foo" = any() {}
            if let 1 = any() {}
            if let 1u32 = any() {}
            if let 1f32 = any() {}
            if let 1.0 = any() {}
            if let true = any() {}
        }
        "#,
        expect![[r#"
            17..28 '{ loop {} }': T
            19..26 'loop {}': !
            24..26 '{}': ()
            37..38 'x': &i32
            46..208 '{     ...) {} }': ()
            52..75 'if let...y() {}': ()
            55..72 'let "f... any()': bool
            59..64 '"foo"': &str
            59..64 '"foo"': &str
            67..70 'any': fn any<&str>() -> &str
            67..72 'any()': &str
            73..75 '{}': ()
            80..99 'if let...y() {}': ()
            83..96 'let 1 = any()': bool
            87..88 '1': i32
            87..88 '1': i32
            91..94 'any': fn any<i32>() -> i32
            91..96 'any()': i32
            97..99 '{}': ()
            104..126 'if let...y() {}': ()
            107..123 'let 1u... any()': bool
            111..115 '1u32': u32
            111..115 '1u32': u32
            118..121 'any': fn any<u32>() -> u32
            118..123 'any()': u32
            124..126 '{}': ()
            131..153 'if let...y() {}': ()
            134..150 'let 1f... any()': bool
            138..142 '1f32': f32
            138..142 '1f32': f32
            145..148 'any': fn any<f32>() -> f32
            145..150 'any()': f32
            151..153 '{}': ()
            158..179 'if let...y() {}': ()
            161..176 'let 1.0 = any()': bool
            165..168 '1.0': f64
            165..168 '1.0': f64
            171..174 'any': fn any<f64>() -> f64
            171..176 'any()': f64
            177..179 '{}': ()
            184..206 'if let...y() {}': ()
            187..203 'let tr... any()': bool
            191..195 'true': bool
            191..195 'true': bool
            198..201 'any': fn any<bool>() -> bool
            198..203 'any()': bool
            204..206 '{}': ()
        "#]],
    );
}

#[test]
fn infer_range_pattern() {
    check_infer_with_mismatches(
        r#"
        fn test(x: &i32) {
            if let 1..76 = 2u32 {}
            if let 1..=76 = 2u32 {}
        }
        "#,
        expect![[r#"
            8..9 'x': &i32
            17..75 '{     ...2 {} }': ()
            23..45 'if let...u32 {}': ()
            26..42 'let 1....= 2u32': bool
            30..35 '1..76': u32
            38..42 '2u32': u32
            43..45 '{}': ()
            50..73 'if let...u32 {}': ()
            53..70 'let 1....= 2u32': bool
            57..63 '1..=76': u32
            66..70 '2u32': u32
            71..73 '{}': ()
        "#]],
    );
}

#[test]
fn infer_pattern_match_ergonomics() {
    check_infer(
        r#"
        struct A<T>(T);

        fn test() {
            let A(n) = &A(1);
            let A(n) = &mut A(1);
        }
        "#,
        expect![[r#"
            27..78 '{     ...(1); }': ()
            37..41 'A(n)': A<i32>
            39..40 'n': &i32
            44..49 '&A(1)': &A<i32>
            45..46 'A': A<i32>(i32) -> A<i32>
            45..49 'A(1)': A<i32>
            47..48 '1': i32
            59..63 'A(n)': A<i32>
            61..62 'n': &mut i32
            66..75 '&mut A(1)': &mut A<i32>
            71..72 'A': A<i32>(i32) -> A<i32>
            71..75 'A(1)': A<i32>
            73..74 '1': i32
        "#]],
    );
}

#[test]
fn infer_pattern_match_ergonomics_ref() {
    cov_mark::check!(match_ergonomics_ref);
    check_infer(
        r#"
        fn test() {
            let v = &(1, &2);
            let (_, &w) = v;
        }
        "#,
        expect![[r#"
            10..56 '{     ...= v; }': ()
            20..21 'v': &(i32, &i32)
            24..32 '&(1, &2)': &(i32, &i32)
            25..32 '(1, &2)': (i32, &i32)
            26..27 '1': i32
            29..31 '&2': &i32
            30..31 '2': i32
            42..49 '(_, &w)': (i32, &i32)
            43..44 '_': i32
            46..48 '&w': &i32
            47..48 'w': i32
            52..53 'v': &(i32, &i32)
        "#]],
    );
}

#[test]
fn infer_pattern_match_slice() {
    check_infer(
        r#"
        fn test() {
            let slice: &[f64] = &[0.0];
            match slice {
                &[] => {},
                &[a] => {
                    a;
                },
                &[b, c] => {
                    b;
                    c;
                }
                _ => {}
            }
        }
        "#,
        expect![[r#"
            10..209 '{     ...   } }': ()
            20..25 'slice': &[f64]
            36..42 '&[0.0]': &[f64; 1]
            37..42 '[0.0]': [f64; 1]
            38..41 '0.0': f64
            48..207 'match ...     }': ()
            54..59 'slice': &[f64]
            70..73 '&[]': &[f64]
            71..73 '[]': [f64]
            77..79 '{}': ()
            89..93 '&[a]': &[f64]
            90..93 '[a]': [f64]
            91..92 'a': f64
            97..123 '{     ...     }': ()
            111..112 'a': f64
            133..140 '&[b, c]': &[f64]
            134..140 '[b, c]': [f64]
            135..136 'b': f64
            138..139 'c': f64
            144..185 '{     ...     }': ()
            158..159 'b': f64
            173..174 'c': f64
            194..195 '_': &[f64]
            199..201 '{}': ()
        "#]],
    );
}

#[test]
fn infer_pattern_match_string_literal() {
    check_infer_with_mismatches(
        r#"
        fn test() {
            let s: &str = "hello";
            match s {
                "hello" => {}
                _ => {}
            }
        }
        "#,
        expect![[r#"
            10..98 '{     ...   } }': ()
            20..21 's': &str
            30..37 '"hello"': &str
            43..96 'match ...     }': ()
            49..50 's': &str
            61..68 '"hello"': &str
            61..68 '"hello"': &str
            72..74 '{}': ()
            83..84 '_': &str
            88..90 '{}': ()
        "#]],
    );
}

#[test]
fn infer_pattern_match_byte_string_literal() {
    check_infer_with_mismatches(
        r#"
        //- minicore: index
        struct S;
        impl<T, const N: usize> core::ops::Index<S> for [T; N] {
            type Output = [u8];
            fn index(&self, index: core::ops::RangeFull) -> &Self::Output {
                loop {}
            }
        }
        fn test(v: [u8; 3]) {
            if let b"foo" = &v[S] {}
            if let b"foo" = &v {}
        }
        "#,
        expect![[r#"
            105..109 'self': &[T; N]
            111..116 'index': {unknown}
            157..180 '{     ...     }': &[u8]
            167..174 'loop {}': !
            172..174 '{}': ()
            191..192 'v': [u8; 3]
            203..261 '{     ...v {} }': ()
            209..233 'if let...[S] {}': ()
            212..230 'let b"... &v[S]': bool
            216..222 'b"foo"': &[u8]
            216..222 'b"foo"': &[u8]
            225..230 '&v[S]': &[u8]
            226..227 'v': [u8; 3]
            226..230 'v[S]': [u8]
            228..229 'S': S
            231..233 '{}': ()
            238..259 'if let... &v {}': ()
            241..256 'let b"foo" = &v': bool
            245..251 'b"foo"': &[u8; 3]
            245..251 'b"foo"': &[u8; 3]
            254..256 '&v': &[u8; 3]
            255..256 'v': [u8; 3]
            257..259 '{}': ()
        "#]],
    );
}

#[test]
fn infer_pattern_match_or() {
    check_infer_with_mismatches(
        r#"
        fn test() {
            let s: &str = "hello";
            match s {
                "hello" | "world" => {}
                _ => {}
            }
        }
        "#,
        expect![[r#"
            10..108 '{     ...   } }': ()
            20..21 's': &str
            30..37 '"hello"': &str
            43..106 'match ...     }': ()
            49..50 's': &str
            61..68 '"hello"': &str
            61..68 '"hello"': &str
            61..78 '"hello...world"': &str
            71..78 '"world"': &str
            71..78 '"world"': &str
            82..84 '{}': ()
            93..94 '_': &str
            98..100 '{}': ()
        "#]],
    );
}

#[test]
fn infer_pattern_match_arr() {
    check_infer(
        r#"
        fn test() {
            let arr: [f64; 2] = [0.0, 1.0];
            match arr {
                [1.0, a] => {
                    a;
                },
                [b, c] => {
                    b;
                    c;
                }
            }
        }
        "#,
        expect![[r#"
            10..179 '{     ...   } }': ()
            20..23 'arr': [f64; 2]
            36..46 '[0.0, 1.0]': [f64; 2]
            37..40 '0.0': f64
            42..45 '1.0': f64
            52..177 'match ...     }': ()
            58..61 'arr': [f64; 2]
            72..80 '[1.0, a]': [f64; 2]
            73..76 '1.0': f64
            73..76 '1.0': f64
            78..79 'a': f64
            84..110 '{     ...     }': ()
            98..99 'a': f64
            120..126 '[b, c]': [f64; 2]
            121..122 'b': f64
            124..125 'c': f64
            130..171 '{     ...     }': ()
            144..145 'b': f64
            159..160 'c': f64
        "#]],
    );
}

#[test]
fn infer_adt_pattern() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            67..288 '{     ...  d; }': ()
            77..78 'e': E
            81..94 'E::A { x: 3 }': E
            91..92 '3': usize
            105..112 'S(y, z)': S
            107..108 'y': u32
            110..111 'z': E
            115..118 'foo': S
            128..147 'E::A {..._var }': E
            138..145 'new_var': usize
            150..151 'e': E
            158..244 'match ...     }': usize
            164..165 'e': E
            176..186 'E::A { x }': E
            183..184 'x': usize
            190..191 'x': usize
            201..205 'E::B': E
            209..212 'foo': bool
            216..217 '1': usize
            227..231 'E::B': E
            235..237 '10': usize
            255..274 'ref d ...{ .. }': &E
            263..274 'E::A { .. }': E
            277..278 'e': E
            284..285 'd': &E
        "#]],
    );
}

#[test]
fn tuple_struct_destructured_with_self() {
    check_infer(
        r#"
struct Foo(usize,);
impl Foo {
    fn f() {
        let Self(s,) = &Foo(0,);
        let Self(s,) = &mut Foo(0,);
        let Self(s,) = Foo(0,);
    }
}
        "#,
        expect![[r#"
            42..151 '{     ...     }': ()
            56..64 'Self(s,)': Foo
            61..62 's': &usize
            67..75 '&Foo(0,)': &Foo
            68..71 'Foo': Foo(usize) -> Foo
            68..75 'Foo(0,)': Foo
            72..73 '0': usize
            89..97 'Self(s,)': Foo
            94..95 's': &mut usize
            100..112 '&mut Foo(0,)': &mut Foo
            105..108 'Foo': Foo(usize) -> Foo
            105..112 'Foo(0,)': Foo
            109..110 '0': usize
            126..134 'Self(s,)': Foo
            131..132 's': usize
            137..140 'Foo': Foo(usize) -> Foo
            137..144 'Foo(0,)': Foo
            141..142 '0': usize
        "#]],
    );
}

#[test]
fn enum_variant_through_self_in_pattern() {
    check_infer(
        r#"
        enum E {
            A { x: usize },
            B(usize),
            C
        }

        impl E {
            fn test() {
                match (loop {}) {
                    Self::A { x } => { x; },
                    Self::B(x) => { x; },
                    Self::C => {},
                };
            }
        }
        "#,
        expect![[r#"
            75..217 '{     ...     }': ()
            85..210 'match ...     }': ()
            92..99 'loop {}': !
            97..99 '{}': ()
            115..128 'Self::A { x }': E
            125..126 'x': usize
            132..138 '{ x; }': ()
            134..135 'x': usize
            152..162 'Self::B(x)': E
            160..161 'x': usize
            166..172 '{ x; }': ()
            168..169 'x': usize
            186..193 'Self::C': E
            197..199 '{}': ()
        "#]],
    );
}

#[test]
fn infer_generics_in_patterns() {
    check_infer(
        r#"
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
        "#,
        expect![[r#"
            78..80 'a1': A<u32>
            90..91 'o': Option<u64>
            106..243 '{     ...  }; }': ()
            116..127 'A { x: x2 }': A<u32>
            123..125 'x2': u32
            130..132 'a1': A<u32>
            142..160 'A::<i6...: x3 }': A<i64>
            156..158 'x3': i64
            163..173 'A { x: 1 }': A<i64>
            170..171 '1': i64
            179..240 'match ...     }': u64
            185..186 'o': Option<u64>
            197..212 'Option::Some(t)': Option<u64>
            210..211 't': u64
            216..217 't': u64
            227..228 '_': Option<u64>
            232..233 '1': u64
        "#]],
    );
}

#[test]
fn infer_const_pattern() {
    check(
        r#"
enum Option<T> { None }
use Option::None;
struct Foo;
const Bar: usize = 1;

fn test() {
    let a: Option<u32> = None;
    let b: Option<i64> = match a {
        None => None,
    };
    let _: () = match () { Foo => () };
                        // ^^^ expected (), got Foo
    let _: () = match () { Bar => () };
                        // ^^^ expected (), got usize
}
        "#,
    );
}

#[test]
fn infer_guard() {
    check_infer(
        r#"
struct S;
impl S { fn foo(&self) -> bool { false } }

fn main() {
    match S {
        s if s.foo() => (),
    }
}
        "#,
        expect![[r#"
            27..31 'self': &S
            41..50 '{ false }': bool
            43..48 'false': bool
            64..115 '{     ...   } }': ()
            70..113 'match ...     }': ()
            76..77 'S': S
            88..89 's': S
            93..94 's': S
            93..100 's.foo()': bool
            104..106 '()': ()
    "#]],
    )
}

#[test]
fn match_ergonomics_in_closure_params() {
    check_infer(
        r#"
//- minicore: fn
fn foo<T, U, F: FnOnce(T) -> U>(t: T, f: F) -> U { loop {} }

fn test() {
    foo(&(1, "a"), |&(x, y)| x); // normal, no match ergonomics
    foo(&(1, "a"), |(x, y)| x);
}
"#,
        expect![[r#"
            32..33 't': T
            38..39 'f': F
            49..60 '{ loop {} }': U
            51..58 'loop {}': !
            56..58 '{}': ()
            72..171 '{     ... x); }': ()
            78..81 'foo': fn foo<&(i32, &str), i32, |&(i32, &str)| -> i32>(&(i32, &str), |&(i32, &str)| -> i32) -> i32
            78..105 'foo(&(...y)| x)': i32
            82..91 '&(1, "a")': &(i32, &str)
            83..91 '(1, "a")': (i32, &str)
            84..85 '1': i32
            87..90 '"a"': &str
            93..104 '|&(x, y)| x': |&(i32, &str)| -> i32
            94..101 '&(x, y)': &(i32, &str)
            95..101 '(x, y)': (i32, &str)
            96..97 'x': i32
            99..100 'y': &str
            103..104 'x': i32
            142..145 'foo': fn foo<&(i32, &str), &i32, |&(i32, &str)| -> &i32>(&(i32, &str), |&(i32, &str)| -> &i32) -> &i32
            142..168 'foo(&(...y)| x)': &i32
            146..155 '&(1, "a")': &(i32, &str)
            147..155 '(1, "a")': (i32, &str)
            148..149 '1': i32
            151..154 '"a"': &str
            157..167 '|(x, y)| x': |&(i32, &str)| -> &i32
            158..164 '(x, y)': (i32, &str)
            159..160 'x': &i32
            162..163 'y': &&str
            166..167 'x': &i32
        "#]],
    );
}

#[test]
fn slice_tail_pattern() {
    check_infer(
        r#"
        fn foo(params: &[i32]) {
            match params {
                [head, tail @ ..] => {
                }
            }
        }
        "#,
        expect![[r#"
            7..13 'params': &[i32]
            23..92 '{     ...   } }': ()
            29..90 'match ...     }': ()
            35..41 'params': &[i32]
            52..69 '[head,... @ ..]': [i32]
            53..57 'head': &i32
            59..68 'tail @ ..': &[i32]
            66..68 '..': [i32]
            73..84 '{         }': ()
        "#]],
    );
}

#[test]
fn box_pattern() {
    check_infer(
        r#"
        pub struct Global;
        #[lang = "owned_box"]
        pub struct Box<T, A = Global>(T);

        fn foo(params: Box<i32>) {
            match params {
                box integer => {}
            }
        }
        "#,
        expect![[r#"
            83..89 'params': Box<i32, Global>
            101..155 '{     ...   } }': ()
            107..153 'match ...     }': ()
            113..119 'params': Box<i32, Global>
            130..141 'box integer': Box<i32, Global>
            134..141 'integer': i32
            145..147 '{}': ()
        "#]],
    );
    check_infer(
        r#"
        #[lang = "owned_box"]
        pub struct Box<T>(T);

        fn foo(params: Box<i32>) {
            match params {
                box integer => {}
            }
        }
        "#,
        expect![[r#"
            52..58 'params': Box<i32>
            70..124 '{     ...   } }': ()
            76..122 'match ...     }': ()
            82..88 'params': Box<i32>
            99..110 'box integer': Box<i32>
            103..110 'integer': i32
            114..116 '{}': ()
        "#]],
    );
}

#[test]
fn tuple_ellipsis_pattern() {
    check_infer_with_mismatches(
        r#"
fn foo(tuple: (u8, i16, f32)) {
    match tuple {
        (.., b, c) => {},
        (a, .., c) => {},
        (a, b, ..) => {},
        (a, b) => {/*too short*/}
        (a, b, c, d) => {/*too long*/}
        _ => {}
    }
}"#,
        expect![[r#"
            7..12 'tuple': (u8, i16, f32)
            30..224 '{     ...   } }': ()
            36..222 'match ...     }': ()
            42..47 'tuple': (u8, i16, f32)
            58..68 '(.., b, c)': (u8, i16, f32)
            63..64 'b': i16
            66..67 'c': f32
            72..74 '{}': ()
            84..94 '(a, .., c)': (u8, i16, f32)
            85..86 'a': u8
            92..93 'c': f32
            98..100 '{}': ()
            110..120 '(a, b, ..)': (u8, i16, f32)
            111..112 'a': u8
            114..115 'b': i16
            124..126 '{}': ()
            136..142 '(a, b)': (u8, i16)
            137..138 'a': u8
            140..141 'b': i16
            146..161 '{/*too short*/}': ()
            170..182 '(a, b, c, d)': (u8, i16, f32, {unknown})
            171..172 'a': u8
            174..175 'b': i16
            177..178 'c': f32
            180..181 'd': {unknown}
            186..200 '{/*too long*/}': ()
            209..210 '_': (u8, i16, f32)
            214..216 '{}': ()
            136..142: expected (u8, i16, f32), got (u8, i16)
            170..182: expected (u8, i16, f32), got (u8, i16, f32, {unknown})
        "#]],
    );
}

#[test]
fn tuple_struct_ellipsis_pattern() {
    check_infer(
        r#"
struct Tuple(u8, i16, f32);
fn foo(tuple: Tuple) {
    match tuple {
        Tuple(.., b, c) => {},
        Tuple(a, .., c) => {},
        Tuple(a, b, ..) => {},
        Tuple(a, b) => {/*too short*/}
        Tuple(a, b, c, d) => {/*too long*/}
        _ => {}
    }
}"#,
        expect![[r#"
            35..40 'tuple': Tuple
            49..268 '{     ...   } }': ()
            55..266 'match ...     }': ()
            61..66 'tuple': Tuple
            77..92 'Tuple(.., b, c)': Tuple
            87..88 'b': i16
            90..91 'c': f32
            96..98 '{}': ()
            108..123 'Tuple(a, .., c)': Tuple
            114..115 'a': u8
            121..122 'c': f32
            127..129 '{}': ()
            139..154 'Tuple(a, b, ..)': Tuple
            145..146 'a': u8
            148..149 'b': i16
            158..160 '{}': ()
            170..181 'Tuple(a, b)': Tuple
            176..177 'a': u8
            179..180 'b': i16
            185..200 '{/*too short*/}': ()
            209..226 'Tuple(... c, d)': Tuple
            215..216 'a': u8
            218..219 'b': i16
            221..222 'c': f32
            224..225 'd': {unknown}
            230..244 '{/*too long*/}': ()
            253..254 '_': Tuple
            258..260 '{}': ()
        "#]],
    );
}

#[test]
fn const_block_pattern() {
    check_infer(
        r#"
struct Foo(usize);
fn foo(foo: Foo) {
    match foo {
        const { Foo(15 + 32) } => {},
        _ => {}
    }
}"#,
        expect![[r#"
            26..29 'foo': Foo
            36..115 '{     ...   } }': ()
            42..113 'match ...     }': ()
            48..51 'foo': Foo
            62..84 'const ... 32) }': Foo
            68..84 '{ Foo(... 32) }': Foo
            70..73 'Foo': Foo(usize) -> Foo
            70..82 'Foo(15 + 32)': Foo
            74..76 '15': usize
            74..81 '15 + 32': usize
            79..81 '32': usize
            88..90 '{}': ()
            100..101 '_': Foo
            105..107 '{}': ()
        "#]],
    );
}

#[test]
fn macro_pat() {
    check_types(
        r#"
macro_rules! pat {
    ($name:ident) => { Enum::Variant1($name) }
}

enum Enum {
    Variant1(u8),
    Variant2,
}

fn f(e: Enum) {
    match e {
        pat!(bind) => {
            bind;
          //^^^^ u8
        }
        Enum::Variant2 => {}
    }
}
    "#,
    )
}

#[test]
fn type_mismatch_in_or_pattern() {
    check_infer_with_mismatches(
        r#"
fn main() {
    match (false,) {
        (true | (),) => {}
        (() | true,) => {}
        (_ | (),) => {}
        (() | _,) => {}
    }
}
"#,
        expect![[r#"
            10..142 '{     ...   } }': ()
            16..140 'match ...     }': ()
            22..30 '(false,)': (bool,)
            23..28 'false': bool
            41..53 '(true | (),)': (bool,)
            42..46 'true': bool
            42..46 'true': bool
            42..51 'true | ()': bool
            49..51 '()': ()
            57..59 '{}': ()
            68..80 '(() | true,)': ((),)
            69..71 '()': ()
            69..78 '() | true': ()
            74..78 'true': bool
            74..78 'true': bool
            84..86 '{}': ()
            95..104 '(_ | (),)': (bool,)
            96..97 '_': bool
            96..102 '_ | ()': bool
            100..102 '()': ()
            108..110 '{}': ()
            119..128 '(() | _,)': ((),)
            120..122 '()': ()
            120..126 '() | _': ()
            125..126 '_': bool
            132..134 '{}': ()
            49..51: expected bool, got ()
            68..80: expected (bool,), got ((),)
            69..71: expected bool, got ()
            69..78: expected bool, got ()
            100..102: expected bool, got ()
            119..128: expected (bool,), got ((),)
            120..122: expected bool, got ()
            120..126: expected bool, got ()
        "#]],
    );
}

#[test]
fn slice_pattern_correctly_handles_array_length() {
    check_infer(
        r#"
fn main() {
    let [head, middle @ .., tail, tail2] = [1, 2, 3, 4, 5];
}
    "#,
        expect![[r#"
            10..73 '{     ... 5]; }': ()
            20..52 '[head,...tail2]': [i32; 5]
            21..25 'head': i32
            27..38 'middle @ ..': [i32; 2]
            36..38 '..': [i32; 2]
            40..44 'tail': i32
            46..51 'tail2': i32
            55..70 '[1, 2, 3, 4, 5]': [i32; 5]
            56..57 '1': i32
            59..60 '2': i32
            62..63 '3': i32
            65..66 '4': i32
            68..69 '5': i32
        "#]],
    );
}

#[test]
fn pattern_lookup_in_value_ns() {
    check_types(
        r#"
use self::Constructor::*;
struct IntRange {
    range: (),
}
enum Constructor {
    IntRange(IntRange),
}
fn main() {
    match Constructor::IntRange(IntRange { range: () }) {
        IntRange(x) => {
            x;
          //^ IntRange
        }
        Constructor::IntRange(x) => {
            x;
          //^ IntRange
        }
    }
}
    "#,
    );
}

#[test]
fn if_let_guards() {
    check_types(
        r#"
fn main() {
    match (0,) {
        opt if let (x,) = opt => {
            x;
          //^ i32
        }
        _ => {}
    }
}
    "#,
    );
}

#[test]
fn tuple_wildcard() {
    check_types(
        r#"
fn main() {
    enum Option<T> {Some(T), None}
    use Option::*;

    let mut x = None;
    x;
  //^ Option<(i32, i32)>

    if let Some((_, _a)) = x {}

    x = Some((1, 2));
}
        "#,
    );
}
