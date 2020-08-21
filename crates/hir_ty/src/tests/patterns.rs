use expect_test::expect;
use test_utils::mark;

use super::{check_infer, check_infer_with_mismatches};

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

            let lambda = |a: u64, b, c: i32| { a + b; c };

            let ref ref_to_x = x;
            let mut mut_x = x;
            let ref mut mut_ref_to_x = x;
            let k = mut_ref_to_x;
        }
        "#,
        expect![[r#"
            8..9 'x': &i32
            17..368 '{     ...o_x; }': ()
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
            164..169 '[val]': [{unknown}]
            165..168 'val': {unknown}
            172..175 'opt': [{unknown}]
            176..204 '{     ...     }': ()
            190..191 'h': {unknown}
            194..197 'val': {unknown}
            214..220 'lambda': |u64, u64, i32| -> i32
            223..255 '|a: u6...b; c }': |u64, u64, i32| -> i32
            224..225 'a': u64
            232..233 'b': u64
            235..236 'c': i32
            243..255 '{ a + b; c }': i32
            245..246 'a': u64
            245..250 'a + b': u64
            249..250 'b': u64
            252..253 'c': i32
            266..278 'ref ref_to_x': &&i32
            281..282 'x': &i32
            292..301 'mut mut_x': &i32
            304..305 'x': &i32
            315..335 'ref mu...f_to_x': &mut &i32
            338..339 'x': &i32
            349..350 'k': &mut &i32
            353..365 'mut_ref_to_x': &mut &i32
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
            59..64 '"foo"': &str
            59..64 '"foo"': &str
            67..70 'any': fn any<&str>() -> &str
            67..72 'any()': &str
            73..75 '{}': ()
            80..99 'if let...y() {}': ()
            87..88 '1': i32
            87..88 '1': i32
            91..94 'any': fn any<i32>() -> i32
            91..96 'any()': i32
            97..99 '{}': ()
            104..126 'if let...y() {}': ()
            111..115 '1u32': u32
            111..115 '1u32': u32
            118..121 'any': fn any<u32>() -> u32
            118..123 'any()': u32
            124..126 '{}': ()
            131..153 'if let...y() {}': ()
            138..142 '1f32': f32
            138..142 '1f32': f32
            145..148 'any': fn any<f32>() -> f32
            145..150 'any()': f32
            151..153 '{}': ()
            158..179 'if let...y() {}': ()
            165..168 '1.0': f64
            165..168 '1.0': f64
            171..174 'any': fn any<f64>() -> f64
            171..176 'any()': f64
            177..179 '{}': ()
            184..206 'if let...y() {}': ()
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
            30..35 '1..76': u32
            38..42 '2u32': u32
            43..45 '{}': ()
            50..73 'if let...u32 {}': ()
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
    mark::check!(match_ergonomics_ref);
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
            36..42 '&[0.0]': &[f64; _]
            37..42 '[0.0]': [f64; _]
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
            20..23 'arr': [f64; _]
            36..46 '[0.0, 1.0]': [f64; _]
            37..40 '0.0': f64
            42..45 '1.0': f64
            52..177 'match ...     }': ()
            58..61 'arr': [f64; _]
            72..80 '[1.0, a]': [f64; _]
            73..76 '1.0': f64
            73..76 '1.0': f64
            78..79 'a': f64
            84..110 '{     ...     }': ()
            98..99 'a': f64
            120..126 '[b, c]': [f64; _]
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
    check_infer_with_mismatches(
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
            let _: () = match () { Foo => Foo }; // Expected mismatch
            let _: () = match () { Bar => Bar }; // Expected mismatch
        }
        "#,
        expect![[r#"
            73..74 '1': usize
            87..309 '{     ...atch }': ()
            97..98 'a': Option<u32>
            114..118 'None': Option<u32>
            128..129 'b': Option<i64>
            145..182 'match ...     }': Option<i64>
            151..152 'a': Option<u32>
            163..167 'None': Option<u32>
            171..175 'None': Option<i64>
            192..193 '_': ()
            200..223 'match ... Foo }': Foo
            206..208 '()': ()
            211..214 'Foo': Foo
            218..221 'Foo': Foo
            254..255 '_': ()
            262..285 'match ... Bar }': usize
            268..270 '()': ()
            273..276 'Bar': usize
            280..283 'Bar': usize
            200..223: expected (), got Foo
            262..285: expected (), got usize
        "#]],
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
        #[lang = "fn_once"]
        trait FnOnce<Args> {
            type Output;
        }

        fn foo<T, U, F: FnOnce(T) -> U>(t: T, f: F) -> U { loop {} }

        fn test() {
            foo(&(1, "a"), |&(x, y)| x); // normal, no match ergonomics
            foo(&(1, "a"), |(x, y)| x);
        }
        "#,
        expect![[r#"
            93..94 't': T
            99..100 'f': F
            110..121 '{ loop {} }': U
            112..119 'loop {}': !
            117..119 '{}': ()
            133..232 '{     ... x); }': ()
            139..142 'foo': fn foo<&(i32, &str), i32, |&(i32, &str)| -> i32>(&(i32, &str), |&(i32, &str)| -> i32) -> i32
            139..166 'foo(&(...y)| x)': i32
            143..152 '&(1, "a")': &(i32, &str)
            144..152 '(1, "a")': (i32, &str)
            145..146 '1': i32
            148..151 '"a"': &str
            154..165 '|&(x, y)| x': |&(i32, &str)| -> i32
            155..162 '&(x, y)': &(i32, &str)
            156..162 '(x, y)': (i32, &str)
            157..158 'x': i32
            160..161 'y': &str
            164..165 'x': i32
            203..206 'foo': fn foo<&(i32, &str), &i32, |&(i32, &str)| -> &i32>(&(i32, &str), |&(i32, &str)| -> &i32) -> &i32
            203..229 'foo(&(...y)| x)': &i32
            207..216 '&(1, "a")': &(i32, &str)
            208..216 '(1, "a")': (i32, &str)
            209..210 '1': i32
            212..215 '"a"': &str
            218..228 '|(x, y)| x': |&(i32, &str)| -> &i32
            219..225 '(x, y)': (i32, &str)
            220..221 'x': &i32
            223..224 'y': &&str
            227..228 'x': &i32
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
