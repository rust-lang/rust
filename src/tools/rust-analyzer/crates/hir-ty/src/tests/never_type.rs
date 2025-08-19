use expect_test::expect;

use super::{check_infer_with_mismatches, check_no_mismatches, check_types};

#[test]
fn infer_never1() {
    check_types(
        r#"
fn test() {
    let t = return;
    t;
} //^ !
"#,
    );
}

#[test]
fn infer_never2() {
    check_types(
        r#"
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { a } else { loop {} };
    a;
} //^ !
"#,
    );
}

#[test]
fn infer_never3() {
    check_types(
        r#"
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { loop {} } else { a };
    a;
  //^ !
}
"#,
    );
}

#[test]
fn never_type_in_generic_args() {
    check_types(
        r#"
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a;
} //^ Option<!>
"#,
    );
}

#[test]
fn never_type_can_be_reinferred1() {
    check_types(
        r#"
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { loop {} } else { a };
    a;
  //^ ()
    if false { a };
}
"#,
    );
}

#[test]
fn never_type_can_be_reinferred2() {
    check_types(
        r#"
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a;
  //^ Option<i32>
    match 42 {
        42 => a,
        _ => Option::Some(42),
    };
}
"#,
    );
}

#[test]
fn never_type_can_be_reinferred3() {
    check_types(
        r#"
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a;
  //^ Option<&'static str>
    match 42 {
        42 => a,
        _ => Option::Some("str"),
    };
}
"#,
    );
}

#[test]
fn match_no_arm() {
    check_types(
        r#"
enum Void {}

fn test(a: Void) {
    let t = match a {};
    t;
} //^ !
"#,
    );
}

#[test]
fn match_unknown_arm() {
    check_types(
        r#"
fn test(a: Option) {
    let t = match 0 {
        _ => unknown,
    };
    t;
} //^ {unknown}
"#,
    );
}

#[test]
fn if_never() {
    check_types(
        r#"
fn test() {
    let i = if true {
        loop {}
    } else {
        3.0
    };
    i;
} //^ f64
"#,
    );
}

#[test]
fn if_else_never() {
    check_types(
        r#"
fn test(input: bool) {
    let i = if input {
        2.0
    } else {
        return
    };
    i;
} //^ f64
"#,
    );
}

#[test]
fn match_first_arm_never() {
    check_types(
        r#"
fn test(a: i32) {
    let i = match a {
        1 => return,
        2 => 2.0,
        3 => loop {},
        _ => 3.0,
    };
    i;
} //^ f64
"#,
    );
}

#[test]
fn match_second_arm_never() {
    check_types(
        r#"
fn test(a: i32) {
    let i = match a {
        1 => 3.0,
        2 => loop {},
        3 => 3.0,
        _ => return,
    };
    i;
} //^ f64
"#,
    );
}

#[test]
fn match_all_arms_never() {
    check_types(
        r#"
fn test(a: i32) {
    let i = match a {
        2 => return,
        _ => loop {},
    };
    i;
} //^ !
"#,
    );
}

#[test]
fn match_no_never_arms() {
    check_types(
        r#"
fn test(a: i32) {
    let i = match a {
        2 => 2.0,
        _ => 3.0,
    };
    i;
} //^ f64
"#,
    );
}

#[test]
fn diverging_expression_1() {
    check_infer_with_mismatches(
        r"
        //- /main.rs
        fn test1() {
            let x: u32 = return;
        }
        fn test2() {
            let x: u32 = { return; };
        }
        fn test3() {
            let x: u32 = loop {};
        }
        fn test4() {
            let x: u32 = { loop {} };
        }
        fn test5() {
            let x: u32 = { if true { loop {}; } else { loop {}; } };
        }
        fn test6() {
            let x: u32 = { let y: u32 = { loop {}; }; };
        }
        ",
        expect![[r"
            11..39 '{     ...urn; }': ()
            21..22 'x': u32
            30..36 'return': !
            51..84 '{     ...; }; }': ()
            61..62 'x': u32
            70..81 '{ return; }': u32
            72..78 'return': !
            96..125 '{     ... {}; }': ()
            106..107 'x': u32
            115..122 'loop {}': !
            120..122 '{}': ()
            137..170 '{     ...} }; }': ()
            147..148 'x': u32
            156..167 '{ loop {} }': u32
            158..165 'loop {}': !
            163..165 '{}': ()
            182..246 '{     ...} }; }': ()
            192..193 'x': u32
            201..243 '{ if t...}; } }': u32
            203..241 'if tru... {}; }': u32
            206..210 'true': bool
            211..223 '{ loop {}; }': u32
            213..220 'loop {}': !
            218..220 '{}': ()
            229..241 '{ loop {}; }': u32
            231..238 'loop {}': !
            236..238 '{}': ()
            258..310 '{     ...; }; }': ()
            268..269 'x': u32
            277..307 '{ let ...; }; }': u32
            283..284 'y': u32
            292..304 '{ loop {}; }': u32
            294..301 'loop {}': !
            299..301 '{}': ()
        "]],
    );
}

#[test]
fn diverging_expression_2() {
    check_infer_with_mismatches(
        r#"
        //- /main.rs
        fn test1() {
            // should give type mismatch
            let x: u32 = { loop {}; "foo" };
        }
        "#,
        expect![[r#"
            11..84 '{     ..." }; }': ()
            54..55 'x': u32
            63..81 '{ loop...foo" }': u32
            65..72 'loop {}': !
            70..72 '{}': ()
            74..79 '"foo"': &'static str
            74..79: expected u32, got &'static str
        "#]],
    );
}

#[test]
fn diverging_expression_3_break() {
    check_infer_with_mismatches(
        r"
        //- minicore: iterator
        //- /main.rs
        fn test1() {
            // should give type mismatch
            let x: u32 = { loop { break; } };
        }
        fn test2() {
            // should give type mismatch
            let x: u32 = { for a in b { break; }; };
            // should give type mismatch as well
            let x: u32 = { for a in b {}; };
            // should give type mismatch as well
            let x: u32 = { for a in b { return; }; };
        }
        fn test3() {
            // should give type mismatch
            let x: u32 = { while true { break; }; };
            // should give type mismatch as well -- there's an implicit break, even if it's never hit
            let x: u32 = { while true {}; };
            // should give type mismatch as well
            let x: u32 = { while true { return; }; };
        }
        ",
        expect![[r#"
            11..85 '{     ...} }; }': ()
            54..55 'x': u32
            63..82 '{ loop...k; } }': u32
            65..80 'loop { break; }': ()
            70..80 '{ break; }': ()
            72..77 'break': !
            65..80: expected u32, got ()
            97..343 '{     ...; }; }': ()
            140..141 'x': u32
            149..175 '{ for ...; }; }': u32
            151..172 'for a ...eak; }': fn into_iter<{unknown}>({unknown}) -> <{unknown} as IntoIterator>::IntoIter
            151..172 'for a ...eak; }': {unknown}
            151..172 'for a ...eak; }': !
            151..172 'for a ...eak; }': {unknown}
            151..172 'for a ...eak; }': &'? mut {unknown}
            151..172 'for a ...eak; }': fn next<{unknown}>(&'? mut {unknown}) -> Option<<{unknown} as Iterator>::Item>
            151..172 'for a ...eak; }': Option<{unknown}>
            151..172 'for a ...eak; }': ()
            151..172 'for a ...eak; }': ()
            151..172 'for a ...eak; }': ()
            151..172 'for a ...eak; }': ()
            155..156 'a': {unknown}
            160..161 'b': {unknown}
            162..172 '{ break; }': ()
            164..169 'break': !
            226..227 'x': u32
            235..253 '{ for ... {}; }': u32
            237..250 'for a in b {}': fn into_iter<{unknown}>({unknown}) -> <{unknown} as IntoIterator>::IntoIter
            237..250 'for a in b {}': {unknown}
            237..250 'for a in b {}': !
            237..250 'for a in b {}': {unknown}
            237..250 'for a in b {}': &'? mut {unknown}
            237..250 'for a in b {}': fn next<{unknown}>(&'? mut {unknown}) -> Option<<{unknown} as Iterator>::Item>
            237..250 'for a in b {}': Option<{unknown}>
            237..250 'for a in b {}': ()
            237..250 'for a in b {}': ()
            237..250 'for a in b {}': ()
            237..250 'for a in b {}': ()
            241..242 'a': {unknown}
            246..247 'b': {unknown}
            248..250 '{}': ()
            304..305 'x': u32
            313..340 '{ for ...; }; }': u32
            315..337 'for a ...urn; }': fn into_iter<{unknown}>({unknown}) -> <{unknown} as IntoIterator>::IntoIter
            315..337 'for a ...urn; }': {unknown}
            315..337 'for a ...urn; }': !
            315..337 'for a ...urn; }': {unknown}
            315..337 'for a ...urn; }': &'? mut {unknown}
            315..337 'for a ...urn; }': fn next<{unknown}>(&'? mut {unknown}) -> Option<<{unknown} as Iterator>::Item>
            315..337 'for a ...urn; }': Option<{unknown}>
            315..337 'for a ...urn; }': ()
            315..337 'for a ...urn; }': ()
            315..337 'for a ...urn; }': ()
            315..337 'for a ...urn; }': ()
            319..320 'a': {unknown}
            324..325 'b': {unknown}
            326..337 '{ return; }': ()
            328..334 'return': !
            149..175: expected u32, got ()
            235..253: expected u32, got ()
            313..340: expected u32, got ()
            355..654 '{     ...; }; }': ()
            398..399 'x': u32
            407..433 '{ whil...; }; }': u32
            409..430 'while ...eak; }': !
            409..430 'while ...eak; }': ()
            409..430 'while ...eak; }': ()
            415..419 'true': bool
            420..430 '{ break; }': ()
            422..427 'break': !
            537..538 'x': u32
            546..564 '{ whil... {}; }': u32
            548..561 'while true {}': !
            548..561 'while true {}': ()
            548..561 'while true {}': ()
            554..558 'true': bool
            559..561 '{}': ()
            615..616 'x': u32
            624..651 '{ whil...; }; }': u32
            626..648 'while ...urn; }': !
            626..648 'while ...urn; }': ()
            626..648 'while ...urn; }': ()
            632..636 'true': bool
            637..648 '{ return; }': ()
            639..645 'return': !
            407..433: expected u32, got ()
            546..564: expected u32, got ()
            624..651: expected u32, got ()
        "#]],
    );
}

#[test]
fn let_else_must_diverge() {
    check_infer_with_mismatches(
        r#"
        fn f() {
            let 1 = 2 else {
                return;
            };
        }
        "#,
        expect![[r#"
            7..54 '{     ...  }; }': ()
            17..18 '1': i32
            17..18 '1': i32
            21..22 '2': i32
            28..51 '{     ...     }': !
            38..44 'return': !
        "#]],
    );
    check_infer_with_mismatches(
        r#"
        fn f() {
            let 1 = 2 else {};
        }
        "#,
        expect![[r#"
            7..33 '{     ... {}; }': ()
            17..18 '1': i32
            17..18 '1': i32
            21..22 '2': i32
            28..30 '{}': !
            28..30: expected !, got ()
        "#]],
    );
}

#[test]
fn issue_11837() {
    check_no_mismatches(
        r#"
//- minicore: result
enum MyErr {
    Err1,
    Err2,
}

fn example_ng() {
    let value: Result<i32, MyErr> = Ok(3);

    loop {
        let ret = match value {
            Ok(value) => value,
            Err(ref err) => {
                match err {
                    MyErr::Err1 => break,
                    MyErr::Err2 => continue,
                };
            }
        };
    }
}
"#,
    );
}

#[test]
fn issue_11814() {
    check_no_mismatches(
        r#"
fn example() -> bool {
    match 1 {
        _ => return true,
    };
}
"#,
    );
}

#[test]
fn reservation_impl_should_be_ignored() {
    // See rust-lang/rust#64631.
    check_types(
        r#"
//- minicore: from
struct S;
#[rustc_reservation_impl]
impl<T> From<!> for T {}
fn foo<T, U: From<T>>(_: U) -> T { loop {} }

fn test() {
    let s = foo(S);
      //^ S
}
"#,
    );
}

#[test]
fn diverging_place_match1() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn not_a_read() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let _: ! = *x;
    }
}
"#,
        expect![[r#"
            21..100 '{     ...   } }': !
            27..98 'unsafe...     }': !
            48..49 'x': *const !
            62..63 '0': i32
            62..68 '0 as _': *const !
            82..83 '_': !
            89..91 '*x': !
            90..91 'x': *const !
            27..98: expected !, got ()
        "#]],
    )
}

#[test]
fn diverging_place_match2() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn not_a_read_implicit() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let _ = *x;
    }
}
"#,
        expect![[r#"
            30..106 '{     ...   } }': !
            36..104 'unsafe...     }': !
            57..58 'x': *const !
            71..72 '0': i32
            71..77 '0 as _': *const !
            91..92 '_': !
            95..97 '*x': !
            96..97 'x': *const !
            36..104: expected !, got ()
        "#]],
    )
}

#[test]
fn diverging_place_match3() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn not_a_read_guide_coercion() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let _: () = *x;
    }
}
"#,
        expect![[r#"
            36..116 '{     ...   } }': !
            42..114 'unsafe...     }': !
            63..64 'x': *const !
            77..78 '0': i32
            77..83 '0 as _': *const !
            97..98 '_': ()
            105..107 '*x': !
            106..107 'x': *const !
            42..114: expected !, got ()
            105..107: expected (), got !
        "#]],
    )
}

#[test]
fn diverging_place_match4() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn empty_match() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        match *x { _ => {} };
    }
}
"#,
        expect![[r#"
            22..108 '{     ...   } }': !
            28..106 'unsafe...     }': !
            49..50 'x': *const !
            63..64 '0': i32
            63..69 '0 as _': *const !
            79..99 'match ...> {} }': ()
            85..87 '*x': !
            86..87 'x': *const !
            90..91 '_': !
            95..97 '{}': ()
            28..106: expected !, got ()
        "#]],
    )
}

#[test]
fn diverging_place_match5() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn field_projection() -> ! {
    unsafe {
        let x: *const (!, ()) = 0 as _;
        let _ = (*x).0;
    }
}
"#,
        expect![[r#"
            27..113 '{     ...   } }': !
            33..111 'unsafe...     }': !
            54..55 'x': *const (!, ())
            74..75 '0': i32
            74..80 '0 as _': *const (!, ())
            94..95 '_': !
            98..104 '(*x).0': !
            99..101 '*x': (!, ())
            100..101 'x': *const (!, ())
            33..111: expected !, got ()
        "#]],
    )
}

#[test]
fn diverging_place_match6() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn covered_arm() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let (_ | 1i32) = *x;
    }
}
"#,
        expect![[r#"
            22..107 '{     ...   } }': !
            28..105 'unsafe...     }': !
            49..50 'x': *const !
            63..64 '0': i32
            63..69 '0 as _': *const !
            84..85 '_': !
            84..92 '_ | 1i32': !
            88..92 '1i32': i32
            88..92 '1i32': i32
            96..98 '*x': !
            97..98 'x': *const !
            28..105: expected !, got ()
            88..92: expected !, got i32
        "#]],
    )
}

#[test]
fn diverging_place_match7() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn uncovered_arm() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let (1i32 | _) = *x;
    }
}
"#,
        expect![[r#"
            24..109 '{     ...   } }': !
            30..107 'unsafe...     }': !
            51..52 'x': *const !
            65..66 '0': i32
            65..71 '0 as _': *const !
            86..90 '1i32': i32
            86..90 '1i32': i32
            86..94 '1i32 | _': !
            93..94 '_': !
            98..100 '*x': !
            99..100 'x': *const !
            30..107: expected !, got ()
            86..90: expected !, got i32
        "#]],
    )
}

#[test]
fn diverging_place_match8() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn coerce_ref_binding() -> ! {
    unsafe {
        let x: *const ! = 0 as _;
        let ref _x: () = *x;
    }
}
"#,
        expect![[r#"
            29..114 '{     ...   } }': !
            35..112 'unsafe...     }': !
            56..57 'x': *const !
            70..71 '0': i32
            70..76 '0 as _': *const !
            90..96 'ref _x': &'? ()
            103..105 '*x': !
            104..105 'x': *const !
            103..105: expected (), got !
        "#]],
    )
}

#[test]
fn never_place_isnt_diverging() {
    check_infer_with_mismatches(
        r#"
//- minicore: sized
fn make_up_a_pointer<T>() -> *const T {
    unsafe {
        let x: *const ! = 0 as _;
        &raw const *x
    }
}
"#,
        expect![[r#"
            38..116 '{     ...   } }': *const T
            44..114 'unsafe...     }': *const T
            65..66 'x': *const !
            79..80 '0': i32
            79..85 '0 as _': *const !
            95..108 '&raw const *x': *const !
            106..108 '*x': !
            107..108 'x': *const !
            95..108: expected *const T, got *const !
        "#]],
    )
}

#[test]
fn diverging_destructuring_assignment() {
    check_infer_with_mismatches(
        r#"
fn foo() {
    let n = match 42 {
        0 => _ = loop {},
        _ => 0,
    };
}
    "#,
        expect![[r#"
            9..84 '{     ...  }; }': ()
            19..20 'n': i32
            23..81 'match ...     }': i32
            29..31 '42': i32
            42..43 '0': i32
            42..43 '0': i32
            47..48 '_': !
            47..58 '_ = loop {}': i32
            51..58 'loop {}': !
            56..58 '{}': ()
            68..69 '_': i32
            73..74 '0': i32
        "#]],
    );
}
