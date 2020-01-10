use insta::assert_snapshot;
use test_utils::covers;

use super::infer;

#[test]
fn bug_484() {
    assert_snapshot!(
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
    [32; 34) '{}': ()
    "###
    );
}

#[test]
fn no_panic_on_field_of_enum() {
    assert_snapshot!(
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
    [32; 44) 'x.some_field': {unknown}
    "###
    );
}

#[test]
fn bug_585() {
    assert_snapshot!(
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
    [78; 80) '()': ()
    "###
    );
}

#[test]
fn bug_651() {
    assert_snapshot!(
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
    [37; 38) 'y': i32
    "###
    );
}

#[test]
fn recursive_vars() {
    covers!(type_var_cycles_resolve_completely);
    covers!(type_var_cycles_resolve_as_possible);
    assert_snapshot!(
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
    [43; 44) 'y': &{unknown}
    "###
    );
}

#[test]
fn recursive_vars_2() {
    covers!(type_var_cycles_resolve_completely);
    covers!(type_var_cycles_resolve_as_possible);
    assert_snapshot!(
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
    [59; 77) '[(x, y..., &x)]': [(&&&{unknown}, &&&{unknown});_]
    [60; 66) '(x, y)': (&&&{unknown}, &&&{unknown})
    [61; 62) 'x': &&{unknown}
    [64; 65) 'y': &&{unknown}
    [68; 76) '(&y, &x)': (&&&{unknown}, &&&{unknown})
    [69; 71) '&y': &&&{unknown}
    [70; 71) 'y': &&{unknown}
    [73; 75) '&x': &&&{unknown}
    [74; 75) 'x': &&{unknown}
    "###
    );
}

#[test]
fn infer_std_crash_1() {
    // caused stack overflow, taken from std
    assert_snapshot!(
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
    [128; 130) '()': ()
    "###
    );
}

#[test]
fn infer_std_crash_2() {
    covers!(type_var_resolves_to_int_var);
    // caused "equating two type variables, ...", taken from std
    assert_snapshot!(
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
    [44; 49) 'b'\n'': u8
    "###
    );
}

#[test]
fn infer_std_crash_3() {
    // taken from rustc
    assert_snapshot!(
        infer(r#"
pub fn compute() {
    match nope!() {
        SizeSkeleton::Pointer { non_zero: true, tail } => {}
    }
}
"#),
        @r###"
    [18; 108) '{     ...   } }': ()
    [24; 106) 'match ...     }': ()
    [30; 37) 'nope!()': {unknown}
    [48; 94) 'SizeSk...tail }': {unknown}
    [82; 86) 'true': {unknown}
    [88; 92) 'tail': {unknown}
    [98; 100) '{}': ()
    "###
    );
}

#[test]
fn infer_std_crash_4() {
    // taken from rustc
    assert_snapshot!(
        infer(r#"
pub fn primitive_type() {
    match *self {
        BorrowedRef { type_: Primitive(p), ..} => {},
    }
}
"#),
        @r###"
    [25; 106) '{     ...   } }': ()
    [31; 104) 'match ...     }': ()
    [37; 42) '*self': {unknown}
    [38; 42) 'self': {unknown}
    [53; 91) 'Borrow...), ..}': {unknown}
    [74; 86) 'Primitive(p)': {unknown}
    [84; 85) 'p': {unknown}
    [95; 97) '{}': ()
    "###
    );
}

#[test]
fn infer_std_crash_5() {
    // taken from rustc
    assert_snapshot!(
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
    [182; 189) 'content': &{unknown}
    [192; 314) 'if ICE...     }': &{unknown}
    [195; 232) 'ICE_RE..._VALUE': {unknown}
    [195; 248) 'ICE_RE...&name)': bool
    [242; 247) '&name': &&&{unknown}
    [243; 247) 'name': &&{unknown}
    [249; 277) '{     ...     }': &&{unknown}
    [263; 267) 'name': &&{unknown}
    [283; 314) '{     ...     }': &{unknown}
    [297; 304) 'content': &{unknown}
    "###
    );
}

#[test]
fn infer_nested_generics_crash() {
    // another crash found typechecking rustc
    assert_snapshot!(
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
    [144; 164) 'query_....value': QueryResponse<R>
    "###
    );
}

#[test]
fn infer_paren_macro_call() {
    assert_snapshot!(
        infer(r#"
macro_rules! bar { () => {0u32} }
fn test() {
    let a = (bar!());
}
"#),
        @r###"
    ![0; 4) '0u32': u32
    [45; 70) '{     ...()); }': ()
    [55; 56) 'a': u32
        "###
    );
}

#[test]
fn bug_1030() {
    assert_snapshot!(infer(r#"
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
    [175; 195) 'FxHash...ault()': HashSet<{unknown}, FxHasher>
    "###
    );
}

#[test]
fn issue_2669() {
    assert_snapshot!(
        infer(
            r#"trait A {}
    trait Write {}
    struct Response<T> {}

    trait D {
        fn foo();
    }

    impl<T:A> D for Response<T> {
        fn foo() {
            end();
            fn end<W: Write>() {
                let _x: T =  loop {};
            }
        }
    }"#
        ),
        @r###"
    [147; 262) '{     ...     }': ()
    [161; 164) 'end': fn end<{unknown}>() -> ()
    [161; 166) 'end()': ()
    [199; 252) '{     ...     }': ()
    [221; 223) '_x': !
    [230; 237) 'loop {}': !
    [235; 237) '{}': ()
    "###
    )
}

#[test]
fn issue_2705() {
    assert_snapshot!(
        infer(r#"
trait Trait {}
fn test() {
    <Trait<u32>>::foo()
}
"#),
        @r###"
    [26; 53) '{     ...oo() }': ()
    [32; 49) '<Trait...>::foo': {unknown}
    [32; 51) '<Trait...:foo()': ()
    "###
    );
}
