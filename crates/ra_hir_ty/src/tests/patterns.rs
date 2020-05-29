use insta::assert_snapshot;
use test_utils::mark;

use super::{infer, infer_with_mismatches};

#[test]
fn infer_pattern() {
    assert_snapshot!(
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
    9..10 'x': &i32
    18..369 '{     ...o_x; }': ()
    28..29 'y': &i32
    32..33 'x': &i32
    43..45 '&z': &i32
    44..45 'z': i32
    48..49 'x': &i32
    59..60 'a': i32
    63..64 'z': i32
    74..80 '(c, d)': (i32, &str)
    75..76 'c': i32
    78..79 'd': &str
    83..95 '(1, "hello")': (i32, &str)
    84..85 '1': i32
    87..94 '"hello"': &str
    102..152 'for (e...     }': ()
    106..112 '(e, f)': ({unknown}, {unknown})
    107..108 'e': {unknown}
    110..111 'f': {unknown}
    116..125 'some_iter': {unknown}
    126..152 '{     ...     }': ()
    140..141 'g': {unknown}
    144..145 'e': {unknown}
    158..205 'if let...     }': ()
    165..170 '[val]': [{unknown}]
    166..169 'val': {unknown}
    173..176 'opt': [{unknown}]
    177..205 '{     ...     }': ()
    191..192 'h': {unknown}
    195..198 'val': {unknown}
    215..221 'lambda': |u64, u64, i32| -> i32
    224..256 '|a: u6...b; c }': |u64, u64, i32| -> i32
    225..226 'a': u64
    233..234 'b': u64
    236..237 'c': i32
    244..256 '{ a + b; c }': i32
    246..247 'a': u64
    246..251 'a + b': u64
    250..251 'b': u64
    253..254 'c': i32
    267..279 'ref ref_to_x': &&i32
    282..283 'x': &i32
    293..302 'mut mut_x': &i32
    305..306 'x': &i32
    316..336 'ref mu...f_to_x': &mut &i32
    339..340 'x': &i32
    350..351 'k': &mut &i32
    354..366 'mut_ref_to_x': &mut &i32
    "###
    );
}

#[test]
fn infer_literal_pattern() {
    assert_snapshot!(
        infer_with_mismatches(r#"
fn any<T>() -> T { loop {} }
fn test(x: &i32) {
    if let "foo" = any() {}
    if let 1 = any() {}
    if let 1u32 = any() {}
    if let 1f32 = any() {}
    if let 1.0 = any() {}
    if let true = any() {}
}
"#, true),
        @r###"
    18..29 '{ loop {} }': T
    20..27 'loop {}': !
    25..27 '{}': ()
    38..39 'x': &i32
    47..209 '{     ...) {} }': ()
    53..76 'if let...y() {}': ()
    60..65 '"foo"': &str
    60..65 '"foo"': &str
    68..71 'any': fn any<&str>() -> &str
    68..73 'any()': &str
    74..76 '{}': ()
    81..100 'if let...y() {}': ()
    88..89 '1': i32
    88..89 '1': i32
    92..95 'any': fn any<i32>() -> i32
    92..97 'any()': i32
    98..100 '{}': ()
    105..127 'if let...y() {}': ()
    112..116 '1u32': u32
    112..116 '1u32': u32
    119..122 'any': fn any<u32>() -> u32
    119..124 'any()': u32
    125..127 '{}': ()
    132..154 'if let...y() {}': ()
    139..143 '1f32': f32
    139..143 '1f32': f32
    146..149 'any': fn any<f32>() -> f32
    146..151 'any()': f32
    152..154 '{}': ()
    159..180 'if let...y() {}': ()
    166..169 '1.0': f64
    166..169 '1.0': f64
    172..175 'any': fn any<f64>() -> f64
    172..177 'any()': f64
    178..180 '{}': ()
    185..207 'if let...y() {}': ()
    192..196 'true': bool
    192..196 'true': bool
    199..202 'any': fn any<bool>() -> bool
    199..204 'any()': bool
    205..207 '{}': ()
    "###
    );
}

#[test]
fn infer_range_pattern() {
    assert_snapshot!(
        infer_with_mismatches(r#"
fn test(x: &i32) {
    if let 1..76 = 2u32 {}
    if let 1..=76 = 2u32 {}
}
"#, true),
        @r###"
    9..10 'x': &i32
    18..76 '{     ...2 {} }': ()
    24..46 'if let...u32 {}': ()
    31..36 '1..76': u32
    39..43 '2u32': u32
    44..46 '{}': ()
    51..74 'if let...u32 {}': ()
    58..64 '1..=76': u32
    67..71 '2u32': u32
    72..74 '{}': ()
    "###
    );
}

#[test]
fn infer_pattern_match_ergonomics() {
    assert_snapshot!(
        infer(r#"
struct A<T>(T);

fn test() {
    let A(n) = &A(1);
    let A(n) = &mut A(1);
}
"#),
    @r###"
    28..79 '{     ...(1); }': ()
    38..42 'A(n)': A<i32>
    40..41 'n': &i32
    45..50 '&A(1)': &A<i32>
    46..47 'A': A<i32>(i32) -> A<i32>
    46..50 'A(1)': A<i32>
    48..49 '1': i32
    60..64 'A(n)': A<i32>
    62..63 'n': &mut i32
    67..76 '&mut A(1)': &mut A<i32>
    72..73 'A': A<i32>(i32) -> A<i32>
    72..76 'A(1)': A<i32>
    74..75 '1': i32
    "###
    );
}

#[test]
fn infer_pattern_match_ergonomics_ref() {
    mark::check!(match_ergonomics_ref);
    assert_snapshot!(
        infer(r#"
fn test() {
    let v = &(1, &2);
    let (_, &w) = v;
}
"#),
    @r###"
    11..57 '{     ...= v; }': ()
    21..22 'v': &(i32, &i32)
    25..33 '&(1, &2)': &(i32, &i32)
    26..33 '(1, &2)': (i32, &i32)
    27..28 '1': i32
    30..32 '&2': &i32
    31..32 '2': i32
    43..50 '(_, &w)': (i32, &i32)
    44..45 '_': i32
    47..49 '&w': &i32
    48..49 'w': i32
    53..54 'v': &(i32, &i32)
    "###
    );
}

#[test]
fn infer_pattern_match_slice() {
    assert_snapshot!(
        infer(r#"
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
"#),
    @r###"
    11..210 '{     ...   } }': ()
    21..26 'slice': &[f64]
    37..43 '&[0.0]': &[f64; _]
    38..43 '[0.0]': [f64; _]
    39..42 '0.0': f64
    49..208 'match ...     }': ()
    55..60 'slice': &[f64]
    71..74 '&[]': &[f64]
    72..74 '[]': [f64]
    78..80 '{}': ()
    90..94 '&[a]': &[f64]
    91..94 '[a]': [f64]
    92..93 'a': f64
    98..124 '{     ...     }': ()
    112..113 'a': f64
    134..141 '&[b, c]': &[f64]
    135..141 '[b, c]': [f64]
    136..137 'b': f64
    139..140 'c': f64
    145..186 '{     ...     }': ()
    159..160 'b': f64
    174..175 'c': f64
    195..196 '_': &[f64]
    200..202 '{}': ()
    "###
    );
}

#[test]
fn infer_pattern_match_arr() {
    assert_snapshot!(
        infer(r#"
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
"#),
    @r###"
    11..180 '{     ...   } }': ()
    21..24 'arr': [f64; _]
    37..47 '[0.0, 1.0]': [f64; _]
    38..41 '0.0': f64
    43..46 '1.0': f64
    53..178 'match ...     }': ()
    59..62 'arr': [f64; _]
    73..81 '[1.0, a]': [f64; _]
    74..77 '1.0': f64
    74..77 '1.0': f64
    79..80 'a': f64
    85..111 '{     ...     }': ()
    99..100 'a': f64
    121..127 '[b, c]': [f64; _]
    122..123 'b': f64
    125..126 'c': f64
    131..172 '{     ...     }': ()
    145..146 'b': f64
    160..161 'c': f64
    "###
    );
}

#[test]
fn infer_adt_pattern() {
    assert_snapshot!(
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
    68..289 '{     ...  d; }': ()
    78..79 'e': E
    82..95 'E::A { x: 3 }': E
    92..93 '3': usize
    106..113 'S(y, z)': S
    108..109 'y': u32
    111..112 'z': E
    116..119 'foo': S
    129..148 'E::A {..._var }': E
    139..146 'new_var': usize
    151..152 'e': E
    159..245 'match ...     }': usize
    165..166 'e': E
    177..187 'E::A { x }': E
    184..185 'x': usize
    191..192 'x': usize
    202..206 'E::B': E
    210..213 'foo': bool
    217..218 '1': usize
    228..232 'E::B': E
    236..238 '10': usize
    256..275 'ref d ...{ .. }': &E
    264..275 'E::A { .. }': E
    278..279 'e': E
    285..286 'd': &E
    "###
    );
}

#[test]
fn enum_variant_through_self_in_pattern() {
    assert_snapshot!(
        infer(r#"
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
"#),
        @r###"
    76..218 '{     ...     }': ()
    86..211 'match ...     }': ()
    93..100 'loop {}': !
    98..100 '{}': ()
    116..129 'Self::A { x }': E
    126..127 'x': usize
    133..139 '{ x; }': ()
    135..136 'x': usize
    153..163 'Self::B(x)': E
    161..162 'x': usize
    167..173 '{ x; }': ()
    169..170 'x': usize
    187..194 'Self::C': E
    198..200 '{}': ()
    "###
    );
}

#[test]
fn infer_generics_in_patterns() {
    assert_snapshot!(
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
    79..81 'a1': A<u32>
    91..92 'o': Option<u64>
    107..244 '{     ...  }; }': ()
    117..128 'A { x: x2 }': A<u32>
    124..126 'x2': u32
    131..133 'a1': A<u32>
    143..161 'A::<i6...: x3 }': A<i64>
    157..159 'x3': i64
    164..174 'A { x: 1 }': A<i64>
    171..172 '1': i64
    180..241 'match ...     }': u64
    186..187 'o': Option<u64>
    198..213 'Option::Some(t)': Option<u64>
    211..212 't': u64
    217..218 't': u64
    228..229 '_': Option<u64>
    233..234 '1': u64
    "###
    );
}

#[test]
fn infer_const_pattern() {
    assert_snapshot!(
        infer_with_mismatches(r#"
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
"#, true),
        @r###"
    74..75 '1': usize
    88..310 '{     ...atch }': ()
    98..99 'a': Option<u32>
    115..119 'None': Option<u32>
    129..130 'b': Option<i64>
    146..183 'match ...     }': Option<i64>
    152..153 'a': Option<u32>
    164..168 'None': Option<u32>
    172..176 'None': Option<i64>
    193..194 '_': ()
    201..224 'match ... Foo }': Foo
    207..209 '()': ()
    212..215 'Foo': Foo
    219..222 'Foo': Foo
    255..256 '_': ()
    263..286 'match ... Bar }': usize
    269..271 '()': ()
    274..277 'Bar': usize
    281..284 'Bar': usize
    201..224: expected (), got Foo
    263..286: expected (), got usize
    "###
    );
}

#[test]
fn infer_guard() {
    assert_snapshot!(
        infer(r#"
struct S;
impl S { fn foo(&self) -> bool { false } }

fn main() {
    match S {
        s if s.foo() => (),
    }
}
    "#), @"
        28..32 'self': &S
        42..51 '{ false }': bool
        44..49 'false': bool
        65..116 '{     ...   } }': ()
        71..114 'match ...     }': ()
        77..78 'S': S
        89..90 's': S
        94..95 's': S
        94..101 's.foo()': bool
        105..107 '()': ()
    ")
}

#[test]
fn match_ergonomics_in_closure_params() {
    assert_snapshot!(
        infer(r#"
#[lang = "fn_once"]
trait FnOnce<Args> {
    type Output;
}

fn foo<T, U, F: FnOnce(T) -> U>(t: T, f: F) -> U { loop {} }

fn test() {
    foo(&(1, "a"), |&(x, y)| x); // normal, no match ergonomics
    foo(&(1, "a"), |(x, y)| x);
}
"#),
        @r###"
    94..95 't': T
    100..101 'f': F
    111..122 '{ loop {} }': U
    113..120 'loop {}': !
    118..120 '{}': ()
    134..233 '{     ... x); }': ()
    140..143 'foo': fn foo<&(i32, &str), i32, |&(i32, &str)| -> i32>(&(i32, &str), |&(i32, &str)| -> i32) -> i32
    140..167 'foo(&(...y)| x)': i32
    144..153 '&(1, "a")': &(i32, &str)
    145..153 '(1, "a")': (i32, &str)
    146..147 '1': i32
    149..152 '"a"': &str
    155..166 '|&(x, y)| x': |&(i32, &str)| -> i32
    156..163 '&(x, y)': &(i32, &str)
    157..163 '(x, y)': (i32, &str)
    158..159 'x': i32
    161..162 'y': &str
    165..166 'x': i32
    204..207 'foo': fn foo<&(i32, &str), &i32, |&(i32, &str)| -> &i32>(&(i32, &str), |&(i32, &str)| -> &i32) -> &i32
    204..230 'foo(&(...y)| x)': &i32
    208..217 '&(1, "a")': &(i32, &str)
    209..217 '(1, "a")': (i32, &str)
    210..211 '1': i32
    213..216 '"a"': &str
    219..229 '|(x, y)| x': |&(i32, &str)| -> &i32
    220..226 '(x, y)': (i32, &str)
    221..222 'x': &i32
    224..225 'y': &&str
    228..229 'x': &i32
    "###
    );
}
