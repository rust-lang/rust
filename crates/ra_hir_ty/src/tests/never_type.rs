use insta::assert_snapshot;

use super::{infer_with_mismatches, type_at};

#[test]
fn infer_never1() {
    let t = type_at(
        r#"
//- /main.rs
fn test() {
    let t = return;
    t<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn infer_never2() {
    let t = type_at(
        r#"
//- /main.rs
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { a } else { loop {} };
    a<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn infer_never3() {
    let t = type_at(
        r#"
//- /main.rs
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { loop {} } else { a };
    a<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn never_type_in_generic_args() {
    let t = type_at(
        r#"
//- /main.rs
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a<|>;
}
"#,
    );
    assert_eq!(t, "Option<!>");
}

#[test]
fn never_type_can_be_reinferred1() {
    let t = type_at(
        r#"
//- /main.rs
fn gen<T>() -> T { loop {} }

fn test() {
    let a = gen();
    if false { loop {} } else { a };
    a<|>;
    if false { a };
}
"#,
    );
    assert_eq!(t, "()");
}

#[test]
fn never_type_can_be_reinferred2() {
    let t = type_at(
        r#"
//- /main.rs
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a<|>;
    match 42 {
        42 => a,
        _ => Option::Some(42),
    };
}
"#,
    );
    assert_eq!(t, "Option<i32>");
}

#[test]
fn never_type_can_be_reinferred3() {
    let t = type_at(
        r#"
//- /main.rs
enum Option<T> { None, Some(T) }

fn test() {
    let a = if true { Option::None } else { Option::Some(return) };
    a<|>;
    match 42 {
        42 => a,
        _ => Option::Some("str"),
    };
}
"#,
    );
    assert_eq!(t, "Option<&str>");
}

#[test]
fn match_no_arm() {
    let t = type_at(
        r#"
//- /main.rs
enum Void {}

fn test(a: Void) {
    let t = match a {};
    t<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn match_unknown_arm() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: Option) {
    let t = match 0 {
        _ => unknown,
    };
    t<|>;
}
"#,
    );
    assert_eq!(t, "{unknown}");
}

#[test]
fn if_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test() {
    let i = if true {
        loop {}
    } else {
        3.0
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn if_else_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(input: bool) {
    let i = if input {
        2.0
    } else {
        return
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn match_first_arm_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        1 => return,
        2 => 2.0,
        3 => loop {},
        _ => 3.0,
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn match_second_arm_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        1 => 3.0,
        2 => loop {},
        3 => 3.0,
        _ => return,
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn match_all_arms_never() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        2 => return,
        _ => loop {},
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "!");
}

#[test]
fn match_no_never_arms() {
    let t = type_at(
        r#"
//- /main.rs
fn test(a: i32) {
    let i = match a {
        2 => 2.0,
        _ => 3.0,
    };
    i<|>;
}
"#,
    );
    assert_eq!(t, "f64");
}

#[test]
fn diverging_expression_1() {
    let t = infer_with_mismatches(
        r#"
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
"#,
        true,
    );
    assert_snapshot!(t, @r###"
    25..53 '{     ...urn; }': ()
    35..36 'x': u32
    44..50 'return': !
    65..98 '{     ...; }; }': ()
    75..76 'x': u32
    84..95 '{ return; }': u32
    86..92 'return': !
    110..139 '{     ... {}; }': ()
    120..121 'x': u32
    129..136 'loop {}': !
    134..136 '{}': ()
    151..184 '{     ...} }; }': ()
    161..162 'x': u32
    170..181 '{ loop {} }': u32
    172..179 'loop {}': !
    177..179 '{}': ()
    196..260 '{     ...} }; }': ()
    206..207 'x': u32
    215..257 '{ if t...}; } }': u32
    217..255 'if tru... {}; }': u32
    220..224 'true': bool
    225..237 '{ loop {}; }': u32
    227..234 'loop {}': !
    232..234 '{}': ()
    243..255 '{ loop {}; }': u32
    245..252 'loop {}': !
    250..252 '{}': ()
    272..324 '{     ...; }; }': ()
    282..283 'x': u32
    291..321 '{ let ...; }; }': u32
    297..298 'y': u32
    306..318 '{ loop {}; }': u32
    308..315 'loop {}': !
    313..315 '{}': ()
    "###);
}

#[test]
fn diverging_expression_2() {
    let t = infer_with_mismatches(
        r#"
//- /main.rs
fn test1() {
    // should give type mismatch
    let x: u32 = { loop {}; "foo" };
}
"#,
        true,
    );
    assert_snapshot!(t, @r###"
    25..98 '{     ..." }; }': ()
    68..69 'x': u32
    77..95 '{ loop...foo" }': &str
    79..86 'loop {}': !
    84..86 '{}': ()
    88..93 '"foo"': &str
    77..95: expected u32, got &str
    88..93: expected u32, got &str
    "###);
}

#[test]
fn diverging_expression_3_break() {
    let t = infer_with_mismatches(
        r#"
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
"#,
        true,
    );
    assert_snapshot!(t, @r###"
    25..99 '{     ...} }; }': ()
    68..69 'x': u32
    77..96 '{ loop...k; } }': ()
    79..94 'loop { break; }': ()
    84..94 '{ break; }': ()
    86..91 'break': !
    77..96: expected u32, got ()
    79..94: expected u32, got ()
    111..357 '{     ...; }; }': ()
    154..155 'x': u32
    163..189 '{ for ...; }; }': ()
    165..186 'for a ...eak; }': ()
    169..170 'a': {unknown}
    174..175 'b': {unknown}
    176..186 '{ break; }': ()
    178..183 'break': !
    240..241 'x': u32
    249..267 '{ for ... {}; }': ()
    251..264 'for a in b {}': ()
    255..256 'a': {unknown}
    260..261 'b': {unknown}
    262..264 '{}': ()
    318..319 'x': u32
    327..354 '{ for ...; }; }': ()
    329..351 'for a ...urn; }': ()
    333..334 'a': {unknown}
    338..339 'b': {unknown}
    340..351 '{ return; }': ()
    342..348 'return': !
    163..189: expected u32, got ()
    249..267: expected u32, got ()
    327..354: expected u32, got ()
    369..668 '{     ...; }; }': ()
    412..413 'x': u32
    421..447 '{ whil...; }; }': ()
    423..444 'while ...eak; }': ()
    429..433 'true': bool
    434..444 '{ break; }': ()
    436..441 'break': !
    551..552 'x': u32
    560..578 '{ whil... {}; }': ()
    562..575 'while true {}': ()
    568..572 'true': bool
    573..575 '{}': ()
    629..630 'x': u32
    638..665 '{ whil...; }; }': ()
    640..662 'while ...urn; }': ()
    646..650 'true': bool
    651..662 '{ return; }': ()
    653..659 'return': !
    421..447: expected u32, got ()
    560..578: expected u32, got ()
    638..665: expected u32, got ()
    "###);
}
