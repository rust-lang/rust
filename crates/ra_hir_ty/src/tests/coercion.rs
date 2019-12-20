use super::infer_with_mismatches;
use insta::assert_snapshot;
use test_utils::covers;

// Infer with some common definitions and impls.
fn infer(source: &str) -> String {
    let defs = r#"
        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
    "#;

    // Append to the end to keep positions unchanged.
    super::infer(&format!("{}{}", source, defs))
}

#[test]
fn infer_block_expr_type_mismatch() {
    assert_snapshot!(
        infer(r#"
fn test() {
    let a: i32 = { 1i64 };
}
"#),
        @r###"
    [11; 41) '{     ...4 }; }': ()
    [21; 22) 'a': i32
    [30; 38) '{ 1i64 }': i64
    [32; 36) '1i64': i64
    "###);
}

#[test]
fn coerce_places() {
    assert_snapshot!(
        infer(r#"
struct S<T> { a: T }

fn f<T>(_: &[T]) -> T { loop {} }
fn g<T>(_: S<&[T]>) -> T { loop {} }

fn gen<T>() -> *mut [T; 2] { loop {} }
fn test1<U>() -> *mut [U] {
    gen()
}

fn test2() {
    let arr: &[u8; 1] = &[1];

    let a: &[_] = arr;
    let b = f(arr);
    let c: &[_] = { arr };
    let d = g(S { a: arr });
    let e: [&[_]; 1] = [arr];
    let f: [&[_]; 2] = [arr; 2];
    let g: (&[_], &[_]) = (arr, arr);
}
"#),
        @r###"
    [31; 32) '_': &[T]
    [45; 56) '{ loop {} }': T
    [47; 54) 'loop {}': !
    [52; 54) '{}': ()
    [65; 66) '_': S<&[T]>
    [82; 93) '{ loop {} }': T
    [84; 91) 'loop {}': !
    [89; 91) '{}': ()
    [122; 133) '{ loop {} }': *mut [T;_]
    [124; 131) 'loop {}': !
    [129; 131) '{}': ()
    [160; 173) '{     gen() }': *mut [U]
    [166; 169) 'gen': fn gen<U>() -> *mut [T;_]
    [166; 171) 'gen()': *mut [U;_]
    [186; 420) '{     ...rr); }': ()
    [196; 199) 'arr': &[u8;_]
    [212; 216) '&[1]': &[u8;_]
    [213; 216) '[1]': [u8;_]
    [214; 215) '1': u8
    [227; 228) 'a': &[u8]
    [237; 240) 'arr': &[u8;_]
    [250; 251) 'b': u8
    [254; 255) 'f': fn f<u8>(&[T]) -> T
    [254; 260) 'f(arr)': u8
    [256; 259) 'arr': &[u8;_]
    [270; 271) 'c': &[u8]
    [280; 287) '{ arr }': &[u8]
    [282; 285) 'arr': &[u8;_]
    [297; 298) 'd': u8
    [301; 302) 'g': fn g<u8>(S<&[T]>) -> T
    [301; 316) 'g(S { a: arr })': u8
    [303; 315) 'S { a: arr }': S<&[u8]>
    [310; 313) 'arr': &[u8;_]
    [326; 327) 'e': [&[u8];_]
    [341; 346) '[arr]': [&[u8];_]
    [342; 345) 'arr': &[u8;_]
    [356; 357) 'f': [&[u8];_]
    [371; 379) '[arr; 2]': [&[u8];_]
    [372; 375) 'arr': &[u8;_]
    [377; 378) '2': usize
    [389; 390) 'g': (&[u8], &[u8])
    [407; 417) '(arr, arr)': (&[u8], &[u8])
    [408; 411) 'arr': &[u8;_]
    [413; 416) 'arr': &[u8;_]
    "###
    );
}

#[test]
fn infer_let_stmt_coerce() {
    assert_snapshot!(
        infer(r#"
fn test() {
    let x: &[i32] = &[1];
}
"#),
        @r###"
    [11; 40) '{     ...[1]; }': ()
    [21; 22) 'x': &[i32]
    [33; 37) '&[1]': &[i32;_]
    [34; 37) '[1]': [i32;_]
    [35; 36) '1': i32
    "###);
}

#[test]
fn infer_custom_coerce_unsized() {
    assert_snapshot!(
        infer(r#"
struct A<T: ?Sized>(*const T);
struct B<T: ?Sized>(*const T);
struct C<T: ?Sized> { inner: *const T }

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<B<U>> for B<T> {}
impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<C<U>> for C<T> {}

fn foo1<T>(x: A<[T]>) -> A<[T]> { x }
fn foo2<T>(x: B<[T]>) -> B<[T]> { x }
fn foo3<T>(x: C<[T]>) -> C<[T]> { x }

fn test(a: A<[u8; 2]>, b: B<[u8; 2]>, c: C<[u8; 2]>) {
    let d = foo1(a);
    let e = foo2(b);
    let f = foo3(c);
}
"#),
        @r###"
    [258; 259) 'x': A<[T]>
    [279; 284) '{ x }': A<[T]>
    [281; 282) 'x': A<[T]>
    [296; 297) 'x': B<[T]>
    [317; 322) '{ x }': B<[T]>
    [319; 320) 'x': B<[T]>
    [334; 335) 'x': C<[T]>
    [355; 360) '{ x }': C<[T]>
    [357; 358) 'x': C<[T]>
    [370; 371) 'a': A<[u8;_]>
    [385; 386) 'b': B<[u8;_]>
    [400; 401) 'c': C<[u8;_]>
    [415; 481) '{     ...(c); }': ()
    [425; 426) 'd': A<[{unknown}]>
    [429; 433) 'foo1': fn foo1<{unknown}>(A<[T]>) -> A<[T]>
    [429; 436) 'foo1(a)': A<[{unknown}]>
    [434; 435) 'a': A<[u8;_]>
    [446; 447) 'e': B<[u8]>
    [450; 454) 'foo2': fn foo2<u8>(B<[T]>) -> B<[T]>
    [450; 457) 'foo2(b)': B<[u8]>
    [455; 456) 'b': B<[u8;_]>
    [467; 468) 'f': C<[u8]>
    [471; 475) 'foo3': fn foo3<u8>(C<[T]>) -> C<[T]>
    [471; 478) 'foo3(c)': C<[u8]>
    [476; 477) 'c': C<[u8;_]>
    "###
    );
}

#[test]
fn infer_if_coerce() {
    assert_snapshot!(
        infer(r#"
fn foo<T>(x: &[T]) -> &[T] { loop {} }
fn test() {
    let x = if true {
        foo(&[1])
    } else {
        &[1]
    };
}
"#),
        @r###"
    [11; 12) 'x': &[T]
    [28; 39) '{ loop {} }': &[T]
    [30; 37) 'loop {}': !
    [35; 37) '{}': ()
    [50; 126) '{     ...  }; }': ()
    [60; 61) 'x': &[i32]
    [64; 123) 'if tru...     }': &[i32]
    [67; 71) 'true': bool
    [72; 97) '{     ...     }': &[i32]
    [82; 85) 'foo': fn foo<i32>(&[T]) -> &[T]
    [82; 91) 'foo(&[1])': &[i32]
    [86; 90) '&[1]': &[i32;_]
    [87; 90) '[1]': [i32;_]
    [88; 89) '1': i32
    [103; 123) '{     ...     }': &[i32;_]
    [113; 117) '&[1]': &[i32;_]
    [114; 117) '[1]': [i32;_]
    [115; 116) '1': i32
    "###
    );
}

#[test]
fn infer_if_else_coerce() {
    assert_snapshot!(
        infer(r#"
fn foo<T>(x: &[T]) -> &[T] { loop {} }
fn test() {
    let x = if true {
        &[1]
    } else {
        foo(&[1])
    };
}
"#),
        @r###"
    [11; 12) 'x': &[T]
    [28; 39) '{ loop {} }': &[T]
    [30; 37) 'loop {}': !
    [35; 37) '{}': ()
    [50; 126) '{     ...  }; }': ()
    [60; 61) 'x': &[i32]
    [64; 123) 'if tru...     }': &[i32]
    [67; 71) 'true': bool
    [72; 92) '{     ...     }': &[i32;_]
    [82; 86) '&[1]': &[i32;_]
    [83; 86) '[1]': [i32;_]
    [84; 85) '1': i32
    [98; 123) '{     ...     }': &[i32]
    [108; 111) 'foo': fn foo<i32>(&[T]) -> &[T]
    [108; 117) 'foo(&[1])': &[i32]
    [112; 116) '&[1]': &[i32;_]
    [113; 116) '[1]': [i32;_]
    [114; 115) '1': i32
    "###
    );
}

#[test]
fn infer_match_first_coerce() {
    assert_snapshot!(
        infer(r#"
fn foo<T>(x: &[T]) -> &[T] { loop {} }
fn test(i: i32) {
    let x = match i {
        2 => foo(&[2]),
        1 => &[1],
        _ => &[3],
    };
}
"#),
        @r###"
    [11; 12) 'x': &[T]
    [28; 39) '{ loop {} }': &[T]
    [30; 37) 'loop {}': !
    [35; 37) '{}': ()
    [48; 49) 'i': i32
    [56; 150) '{     ...  }; }': ()
    [66; 67) 'x': &[i32]
    [70; 147) 'match ...     }': &[i32]
    [76; 77) 'i': i32
    [88; 89) '2': i32
    [93; 96) 'foo': fn foo<i32>(&[T]) -> &[T]
    [93; 102) 'foo(&[2])': &[i32]
    [97; 101) '&[2]': &[i32;_]
    [98; 101) '[2]': [i32;_]
    [99; 100) '2': i32
    [112; 113) '1': i32
    [117; 121) '&[1]': &[i32;_]
    [118; 121) '[1]': [i32;_]
    [119; 120) '1': i32
    [131; 132) '_': i32
    [136; 140) '&[3]': &[i32;_]
    [137; 140) '[3]': [i32;_]
    [138; 139) '3': i32
    "###
    );
}

#[test]
fn infer_match_second_coerce() {
    assert_snapshot!(
        infer(r#"
fn foo<T>(x: &[T]) -> &[T] { loop {} }
fn test(i: i32) {
    let x = match i {
        1 => &[1],
        2 => foo(&[2]),
        _ => &[3],
    };
}
"#),
        @r###"
    [11; 12) 'x': &[T]
    [28; 39) '{ loop {} }': &[T]
    [30; 37) 'loop {}': !
    [35; 37) '{}': ()
    [48; 49) 'i': i32
    [56; 150) '{     ...  }; }': ()
    [66; 67) 'x': &[i32]
    [70; 147) 'match ...     }': &[i32]
    [76; 77) 'i': i32
    [88; 89) '1': i32
    [93; 97) '&[1]': &[i32;_]
    [94; 97) '[1]': [i32;_]
    [95; 96) '1': i32
    [107; 108) '2': i32
    [112; 115) 'foo': fn foo<i32>(&[T]) -> &[T]
    [112; 121) 'foo(&[2])': &[i32]
    [116; 120) '&[2]': &[i32;_]
    [117; 120) '[2]': [i32;_]
    [118; 119) '2': i32
    [131; 132) '_': i32
    [136; 140) '&[3]': &[i32;_]
    [137; 140) '[3]': [i32;_]
    [138; 139) '3': i32
    "###
    );
}

#[test]
fn coerce_merge_one_by_one1() {
    covers!(coerce_merge_fail_fallback);

    assert_snapshot!(
        infer(r#"
fn test() {
    let t = &mut 1;
    let x = match 1 {
        1 => t as *mut i32,
        2 => t as &i32,
        _ => t as *const i32,
    };
}
"#),
        @r###"
    [11; 145) '{     ...  }; }': ()
    [21; 22) 't': &mut i32
    [25; 31) '&mut 1': &mut i32
    [30; 31) '1': i32
    [41; 42) 'x': *const i32
    [45; 142) 'match ...     }': *const i32
    [51; 52) '1': i32
    [63; 64) '1': i32
    [68; 69) 't': &mut i32
    [68; 81) 't as *mut i32': *mut i32
    [91; 92) '2': i32
    [96; 97) 't': &mut i32
    [96; 105) 't as &i32': &i32
    [115; 116) '_': i32
    [120; 121) 't': &mut i32
    [120; 135) 't as *const i32': *const i32
    "###
    );
}

#[test]
fn return_coerce_unknown() {
    assert_snapshot!(
        infer_with_mismatches(r#"
fn foo() -> u32 {
    return unknown;
}
"#, true),
        @r###"
    [17; 40) '{     ...own; }': !
    [23; 37) 'return unknown': !
    [30; 37) 'unknown': u32
    "###
    );
}

#[test]
fn coerce_autoderef() {
    assert_snapshot!(
        infer_with_mismatches(r#"
struct Foo;
fn takes_ref_foo(x: &Foo) {}
fn test() {
    takes_ref_foo(&Foo);
    takes_ref_foo(&&Foo);
    takes_ref_foo(&&&Foo);
}
"#, true),
        @r###"
    [30; 31) 'x': &Foo
    [39; 41) '{}': ()
    [52; 133) '{     ...oo); }': ()
    [58; 71) 'takes_ref_foo': fn takes_ref_foo(&Foo) -> ()
    [58; 77) 'takes_...(&Foo)': ()
    [72; 76) '&Foo': &Foo
    [73; 76) 'Foo': Foo
    [83; 96) 'takes_ref_foo': fn takes_ref_foo(&Foo) -> ()
    [83; 103) 'takes_...&&Foo)': ()
    [97; 102) '&&Foo': &&Foo
    [98; 102) '&Foo': &Foo
    [99; 102) 'Foo': Foo
    [109; 122) 'takes_ref_foo': fn takes_ref_foo(&Foo) -> ()
    [109; 130) 'takes_...&&Foo)': ()
    [123; 129) '&&&Foo': &&&Foo
    [124; 129) '&&Foo': &&Foo
    [125; 129) '&Foo': &Foo
    [126; 129) 'Foo': Foo
    "###
    );
}

#[test]
fn coerce_autoderef_generic() {
    assert_snapshot!(
        infer_with_mismatches(r#"
struct Foo;
fn takes_ref<T>(x: &T) -> T { *x }
fn test() {
    takes_ref(&Foo);
    takes_ref(&&Foo);
    takes_ref(&&&Foo);
}
"#, true),
        @r###"
    [29; 30) 'x': &T
    [41; 47) '{ *x }': T
    [43; 45) '*x': T
    [44; 45) 'x': &T
    [58; 127) '{     ...oo); }': ()
    [64; 73) 'takes_ref': fn takes_ref<Foo>(&T) -> T
    [64; 79) 'takes_ref(&Foo)': Foo
    [74; 78) '&Foo': &Foo
    [75; 78) 'Foo': Foo
    [85; 94) 'takes_ref': fn takes_ref<&Foo>(&T) -> T
    [85; 101) 'takes_...&&Foo)': &Foo
    [95; 100) '&&Foo': &&Foo
    [96; 100) '&Foo': &Foo
    [97; 100) 'Foo': Foo
    [107; 116) 'takes_ref': fn takes_ref<&&Foo>(&T) -> T
    [107; 124) 'takes_...&&Foo)': &&Foo
    [117; 123) '&&&Foo': &&&Foo
    [118; 123) '&&Foo': &&Foo
    [119; 123) '&Foo': &Foo
    [120; 123) 'Foo': Foo
    "###
    );
}

#[test]
fn closure_return_coerce() {
    assert_snapshot!(
        infer_with_mismatches(r#"
fn foo() {
    let x = || {
        if true {
            return &1u32;
        }
        &&1u32
    };
}
"#, true),
        @r###"
    [10; 106) '{     ...  }; }': ()
    [20; 21) 'x': || -> &u32
    [24; 103) '|| {  ...     }': || -> &u32
    [27; 103) '{     ...     }': &u32
    [37; 82) 'if tru...     }': ()
    [40; 44) 'true': bool
    [45; 82) '{     ...     }': !
    [59; 71) 'return &1u32': !
    [66; 71) '&1u32': &u32
    [67; 71) '1u32': u32
    [91; 97) '&&1u32': &&u32
    [92; 97) '&1u32': &u32
    [93; 97) '1u32': u32
    "###
    );
}

#[test]
fn coerce_fn_item_to_fn_ptr() {
    assert_snapshot!(
        infer_with_mismatches(r#"
fn foo(x: u32) -> isize { 1 }
fn test() {
    let f: fn(u32) -> isize = foo;
}
"#, true),
        @r###"
    [8; 9) 'x': u32
    [25; 30) '{ 1 }': isize
    [27; 28) '1': isize
    [41; 79) '{     ...foo; }': ()
    [51; 52) 'f': fn(u32) -> isize
    [73; 76) 'foo': fn foo(u32) -> isize
    "###
    );
}

#[test]
fn coerce_closure_to_fn_ptr() {
    assert_snapshot!(
        infer_with_mismatches(r#"
fn test() {
    let f: fn(u32) -> isize = |x| { 1 };
}
"#, true),
        @r###"
    [11; 55) '{     ...1 }; }': ()
    [21; 22) 'f': fn(u32) -> isize
    [43; 52) '|x| { 1 }': |u32| -> isize
    [44; 45) 'x': u32
    [47; 52) '{ 1 }': isize
    [49; 50) '1': isize
    "###
    );
}
