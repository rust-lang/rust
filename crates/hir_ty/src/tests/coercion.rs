use expect_test::expect;
use test_utils::mark;

use super::{check_infer, check_infer_with_mismatches};

#[test]
fn infer_block_expr_type_mismatch() {
    check_infer(
        r"
        fn test() {
            let a: i32 = { 1i64 };
        }
        ",
        expect![[r"
            10..40 '{     ...4 }; }': ()
            20..21 'a': i32
            29..37 '{ 1i64 }': i64
            31..35 '1i64': i64
        "]],
    );
}

#[test]
fn coerce_places() {
    check_infer(
        r#"
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

        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
        "#,
        expect![[r"
            30..31 '_': &[T]
            44..55 '{ loop {} }': T
            46..53 'loop {}': !
            51..53 '{}': ()
            64..65 '_': S<&[T]>
            81..92 '{ loop {} }': T
            83..90 'loop {}': !
            88..90 '{}': ()
            121..132 '{ loop {} }': *mut [T; _]
            123..130 'loop {}': !
            128..130 '{}': ()
            159..172 '{     gen() }': *mut [U]
            165..168 'gen': fn gen<U>() -> *mut [U; _]
            165..170 'gen()': *mut [U; _]
            185..419 '{     ...rr); }': ()
            195..198 'arr': &[u8; _]
            211..215 '&[1]': &[u8; _]
            212..215 '[1]': [u8; _]
            213..214 '1': u8
            226..227 'a': &[u8]
            236..239 'arr': &[u8; _]
            249..250 'b': u8
            253..254 'f': fn f<u8>(&[u8]) -> u8
            253..259 'f(arr)': u8
            255..258 'arr': &[u8; _]
            269..270 'c': &[u8]
            279..286 '{ arr }': &[u8]
            281..284 'arr': &[u8; _]
            296..297 'd': u8
            300..301 'g': fn g<u8>(S<&[u8]>) -> u8
            300..315 'g(S { a: arr })': u8
            302..314 'S { a: arr }': S<&[u8]>
            309..312 'arr': &[u8; _]
            325..326 'e': [&[u8]; _]
            340..345 '[arr]': [&[u8]; _]
            341..344 'arr': &[u8; _]
            355..356 'f': [&[u8]; _]
            370..378 '[arr; 2]': [&[u8]; _]
            371..374 'arr': &[u8; _]
            376..377 '2': usize
            388..389 'g': (&[u8], &[u8])
            406..416 '(arr, arr)': (&[u8], &[u8])
            407..410 'arr': &[u8; _]
            412..415 'arr': &[u8; _]
        "]],
    );
}

#[test]
fn infer_let_stmt_coerce() {
    check_infer(
        r"
        fn test() {
            let x: &[isize] = &[1];
            let x: *const [isize] = &[1];
        }
        ",
        expect![[r"
            10..75 '{     ...[1]; }': ()
            20..21 'x': &[isize]
            34..38 '&[1]': &[isize; _]
            35..38 '[1]': [isize; _]
            36..37 '1': isize
            48..49 'x': *const [isize]
            68..72 '&[1]': &[isize; _]
            69..72 '[1]': [isize; _]
            70..71 '1': isize
        "]],
    );
}

#[test]
fn infer_custom_coerce_unsized() {
    check_infer(
        r#"
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


        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
        "#,
        expect![[r"
            257..258 'x': A<[T]>
            278..283 '{ x }': A<[T]>
            280..281 'x': A<[T]>
            295..296 'x': B<[T]>
            316..321 '{ x }': B<[T]>
            318..319 'x': B<[T]>
            333..334 'x': C<[T]>
            354..359 '{ x }': C<[T]>
            356..357 'x': C<[T]>
            369..370 'a': A<[u8; _]>
            384..385 'b': B<[u8; _]>
            399..400 'c': C<[u8; _]>
            414..480 '{     ...(c); }': ()
            424..425 'd': A<[{unknown}]>
            428..432 'foo1': fn foo1<{unknown}>(A<[{unknown}]>) -> A<[{unknown}]>
            428..435 'foo1(a)': A<[{unknown}]>
            433..434 'a': A<[u8; _]>
            445..446 'e': B<[u8]>
            449..453 'foo2': fn foo2<u8>(B<[u8]>) -> B<[u8]>
            449..456 'foo2(b)': B<[u8]>
            454..455 'b': B<[u8; _]>
            466..467 'f': C<[u8]>
            470..474 'foo3': fn foo3<u8>(C<[u8]>) -> C<[u8]>
            470..477 'foo3(c)': C<[u8]>
            475..476 'c': C<[u8; _]>
        "]],
    );
}

#[test]
fn infer_if_coerce() {
    check_infer(
        r#"
        fn foo<T>(x: &[T]) -> &[T] { loop {} }
        fn test() {
            let x = if true {
                foo(&[1])
            } else {
                &[1]
            };
        }


        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        "#,
        expect![[r"
            10..11 'x': &[T]
            27..38 '{ loop {} }': &[T]
            29..36 'loop {}': !
            34..36 '{}': ()
            49..125 '{     ...  }; }': ()
            59..60 'x': &[i32]
            63..122 'if tru...     }': &[i32]
            66..70 'true': bool
            71..96 '{     ...     }': &[i32]
            81..84 'foo': fn foo<i32>(&[i32]) -> &[i32]
            81..90 'foo(&[1])': &[i32]
            85..89 '&[1]': &[i32; _]
            86..89 '[1]': [i32; _]
            87..88 '1': i32
            102..122 '{     ...     }': &[i32; _]
            112..116 '&[1]': &[i32; _]
            113..116 '[1]': [i32; _]
            114..115 '1': i32
        "]],
    );
}

#[test]
fn infer_if_else_coerce() {
    check_infer(
        r#"
        fn foo<T>(x: &[T]) -> &[T] { loop {} }
        fn test() {
            let x = if true {
                &[1]
            } else {
                foo(&[1])
            };
        }

        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
        "#,
        expect![[r"
            10..11 'x': &[T]
            27..38 '{ loop {} }': &[T]
            29..36 'loop {}': !
            34..36 '{}': ()
            49..125 '{     ...  }; }': ()
            59..60 'x': &[i32]
            63..122 'if tru...     }': &[i32]
            66..70 'true': bool
            71..91 '{     ...     }': &[i32; _]
            81..85 '&[1]': &[i32; _]
            82..85 '[1]': [i32; _]
            83..84 '1': i32
            97..122 '{     ...     }': &[i32]
            107..110 'foo': fn foo<i32>(&[i32]) -> &[i32]
            107..116 'foo(&[1])': &[i32]
            111..115 '&[1]': &[i32; _]
            112..115 '[1]': [i32; _]
            113..114 '1': i32
        "]],
    )
}

#[test]
fn infer_match_first_coerce() {
    check_infer(
        r#"
        fn foo<T>(x: &[T]) -> &[T] { loop {} }
        fn test(i: i32) {
            let x = match i {
                2 => foo(&[2]),
                1 => &[1],
                _ => &[3],
            };
        }

        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        "#,
        expect![[r"
            10..11 'x': &[T]
            27..38 '{ loop {} }': &[T]
            29..36 'loop {}': !
            34..36 '{}': ()
            47..48 'i': i32
            55..149 '{     ...  }; }': ()
            65..66 'x': &[i32]
            69..146 'match ...     }': &[i32]
            75..76 'i': i32
            87..88 '2': i32
            87..88 '2': i32
            92..95 'foo': fn foo<i32>(&[i32]) -> &[i32]
            92..101 'foo(&[2])': &[i32]
            96..100 '&[2]': &[i32; _]
            97..100 '[2]': [i32; _]
            98..99 '2': i32
            111..112 '1': i32
            111..112 '1': i32
            116..120 '&[1]': &[i32; _]
            117..120 '[1]': [i32; _]
            118..119 '1': i32
            130..131 '_': i32
            135..139 '&[3]': &[i32; _]
            136..139 '[3]': [i32; _]
            137..138 '3': i32
    "]],
    );
}

#[test]
fn infer_match_second_coerce() {
    check_infer(
        r#"
        fn foo<T>(x: &[T]) -> &[T] { loop {} }
        fn test(i: i32) {
            let x = match i {
                1 => &[1],
                2 => foo(&[2]),
                _ => &[3],
            };
        }

        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T: ?Sized> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<'a, 'b: 'a, T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<&'a U> for &'b T {}
        impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<*mut U> for *mut T {}
        "#,
        expect![[r"
            10..11 'x': &[T]
            27..38 '{ loop {} }': &[T]
            29..36 'loop {}': !
            34..36 '{}': ()
            47..48 'i': i32
            55..149 '{     ...  }; }': ()
            65..66 'x': &[i32]
            69..146 'match ...     }': &[i32]
            75..76 'i': i32
            87..88 '1': i32
            87..88 '1': i32
            92..96 '&[1]': &[i32; _]
            93..96 '[1]': [i32; _]
            94..95 '1': i32
            106..107 '2': i32
            106..107 '2': i32
            111..114 'foo': fn foo<i32>(&[i32]) -> &[i32]
            111..120 'foo(&[2])': &[i32]
            115..119 '&[2]': &[i32; _]
            116..119 '[2]': [i32; _]
            117..118 '2': i32
            130..131 '_': i32
            135..139 '&[3]': &[i32; _]
            136..139 '[3]': [i32; _]
            137..138 '3': i32
    "]],
    );
}

#[test]
fn coerce_merge_one_by_one1() {
    mark::check!(coerce_merge_fail_fallback);

    check_infer(
        r"
        fn test() {
            let t = &mut 1;
            let x = match 1 {
                1 => t as *mut i32,
                2 => t as &i32,
                _ => t as *const i32,
            };
        }
        ",
        expect![[r"
            10..144 '{     ...  }; }': ()
            20..21 't': &mut i32
            24..30 '&mut 1': &mut i32
            29..30 '1': i32
            40..41 'x': *const i32
            44..141 'match ...     }': *const i32
            50..51 '1': i32
            62..63 '1': i32
            62..63 '1': i32
            67..68 't': &mut i32
            67..80 't as *mut i32': *mut i32
            90..91 '2': i32
            90..91 '2': i32
            95..96 't': &mut i32
            95..104 't as &i32': &i32
            114..115 '_': i32
            119..120 't': &mut i32
            119..134 't as *const i32': *const i32
    "]],
    );
}

#[test]
fn return_coerce_unknown() {
    check_infer_with_mismatches(
        r"
        fn foo() -> u32 {
            return unknown;
        }
        ",
        expect![[r"
            16..39 '{     ...own; }': u32
            22..36 'return unknown': !
            29..36 'unknown': u32
        "]],
    );
}

#[test]
fn coerce_autoderef() {
    check_infer_with_mismatches(
        r"
        struct Foo;
        fn takes_ref_foo(x: &Foo) {}
        fn test() {
            takes_ref_foo(&Foo);
            takes_ref_foo(&&Foo);
            takes_ref_foo(&&&Foo);
        }
        ",
        expect![[r"
            29..30 'x': &Foo
            38..40 '{}': ()
            51..132 '{     ...oo); }': ()
            57..70 'takes_ref_foo': fn takes_ref_foo(&Foo)
            57..76 'takes_...(&Foo)': ()
            71..75 '&Foo': &Foo
            72..75 'Foo': Foo
            82..95 'takes_ref_foo': fn takes_ref_foo(&Foo)
            82..102 'takes_...&&Foo)': ()
            96..101 '&&Foo': &&Foo
            97..101 '&Foo': &Foo
            98..101 'Foo': Foo
            108..121 'takes_ref_foo': fn takes_ref_foo(&Foo)
            108..129 'takes_...&&Foo)': ()
            122..128 '&&&Foo': &&&Foo
            123..128 '&&Foo': &&Foo
            124..128 '&Foo': &Foo
            125..128 'Foo': Foo
        "]],
    );
}

#[test]
fn coerce_autoderef_generic() {
    check_infer_with_mismatches(
        r"
        struct Foo;
        fn takes_ref<T>(x: &T) -> T { *x }
        fn test() {
            takes_ref(&Foo);
            takes_ref(&&Foo);
            takes_ref(&&&Foo);
        }
        ",
        expect![[r"
            28..29 'x': &T
            40..46 '{ *x }': T
            42..44 '*x': T
            43..44 'x': &T
            57..126 '{     ...oo); }': ()
            63..72 'takes_ref': fn takes_ref<Foo>(&Foo) -> Foo
            63..78 'takes_ref(&Foo)': Foo
            73..77 '&Foo': &Foo
            74..77 'Foo': Foo
            84..93 'takes_ref': fn takes_ref<&Foo>(&&Foo) -> &Foo
            84..100 'takes_...&&Foo)': &Foo
            94..99 '&&Foo': &&Foo
            95..99 '&Foo': &Foo
            96..99 'Foo': Foo
            106..115 'takes_ref': fn takes_ref<&&Foo>(&&&Foo) -> &&Foo
            106..123 'takes_...&&Foo)': &&Foo
            116..122 '&&&Foo': &&&Foo
            117..122 '&&Foo': &&Foo
            118..122 '&Foo': &Foo
            119..122 'Foo': Foo
        "]],
    );
}

#[test]
fn coerce_autoderef_block() {
    check_infer_with_mismatches(
        r#"
        struct String {}
        #[lang = "deref"]
        trait Deref { type Target; }
        impl Deref for String { type Target = str; }
        fn takes_ref_str(x: &str) {}
        fn returns_string() -> String { loop {} }
        fn test() {
            takes_ref_str(&{ returns_string() });
        }
        "#,
        expect![[r"
            126..127 'x': &str
            135..137 '{}': ()
            168..179 '{ loop {} }': String
            170..177 'loop {}': !
            175..177 '{}': ()
            190..235 '{     ... }); }': ()
            196..209 'takes_ref_str': fn takes_ref_str(&str)
            196..232 'takes_...g() })': ()
            210..231 '&{ ret...ng() }': &String
            211..231 '{ retu...ng() }': String
            213..227 'returns_string': fn returns_string() -> String
            213..229 'return...ring()': String
        "]],
    );
}

#[test]
fn closure_return_coerce() {
    check_infer_with_mismatches(
        r"
        fn foo() {
            let x = || {
                if true {
                    return &1u32;
                }
                &&1u32
            };
        }
        ",
        expect![[r"
            9..105 '{     ...  }; }': ()
            19..20 'x': || -> &u32
            23..102 '|| {  ...     }': || -> &u32
            26..102 '{     ...     }': &u32
            36..81 'if tru...     }': ()
            39..43 'true': bool
            44..81 '{     ...     }': ()
            58..70 'return &1u32': !
            65..70 '&1u32': &u32
            66..70 '1u32': u32
            90..96 '&&1u32': &&u32
            91..96 '&1u32': &u32
            92..96 '1u32': u32
        "]],
    );
}

#[test]
fn coerce_fn_item_to_fn_ptr() {
    check_infer_with_mismatches(
        r"
        fn foo(x: u32) -> isize { 1 }
        fn test() {
            let f: fn(u32) -> isize = foo;
        }
        ",
        expect![[r"
            7..8 'x': u32
            24..29 '{ 1 }': isize
            26..27 '1': isize
            40..78 '{     ...foo; }': ()
            50..51 'f': fn(u32) -> isize
            72..75 'foo': fn foo(u32) -> isize
        "]],
    );
}

#[test]
fn coerce_fn_items_in_match_arms() {
    mark::check!(coerce_fn_reification);

    check_infer_with_mismatches(
        r"
        fn foo1(x: u32) -> isize { 1 }
        fn foo2(x: u32) -> isize { 2 }
        fn foo3(x: u32) -> isize { 3 }
        fn test() {
            let x = match 1 {
                1 => foo1,
                2 => foo2,
                _ => foo3,
            };
        }
        ",
        expect![[r"
            8..9 'x': u32
            25..30 '{ 1 }': isize
            27..28 '1': isize
            39..40 'x': u32
            56..61 '{ 2 }': isize
            58..59 '2': isize
            70..71 'x': u32
            87..92 '{ 3 }': isize
            89..90 '3': isize
            103..192 '{     ...  }; }': ()
            113..114 'x': fn(u32) -> isize
            117..189 'match ...     }': fn(u32) -> isize
            123..124 '1': i32
            135..136 '1': i32
            135..136 '1': i32
            140..144 'foo1': fn foo1(u32) -> isize
            154..155 '2': i32
            154..155 '2': i32
            159..163 'foo2': fn foo2(u32) -> isize
            173..174 '_': i32
            178..182 'foo3': fn foo3(u32) -> isize
        "]],
    );
}

#[test]
fn coerce_closure_to_fn_ptr() {
    check_infer_with_mismatches(
        r"
        fn test() {
            let f: fn(u32) -> isize = |x| { 1 };
        }
        ",
        expect![[r"
            10..54 '{     ...1 }; }': ()
            20..21 'f': fn(u32) -> isize
            42..51 '|x| { 1 }': |u32| -> isize
            43..44 'x': u32
            46..51 '{ 1 }': isize
            48..49 '1': isize
        "]],
    );
}

#[test]
fn coerce_placeholder_ref() {
    // placeholders should unify, even behind references
    check_infer_with_mismatches(
        r"
        struct S<T> { t: T }
        impl<TT> S<TT> {
            fn get(&self) -> &TT {
                &self.t
            }
        }
        ",
        expect![[r"
            50..54 'self': &S<TT>
            63..86 '{     ...     }': &TT
            73..80 '&self.t': &TT
            74..78 'self': &S<TT>
            74..80 'self.t': TT
        "]],
    );
}

#[test]
fn coerce_unsize_array() {
    check_infer_with_mismatches(
        r#"
        #[lang = "unsize"]
        pub trait Unsize<T> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<T: Unsize<U>, U> CoerceUnsized<&U> for &T {}

        fn test() {
            let f: &[usize] = &[1, 2, 3];
        }
        "#,
        expect![[r"
            161..198 '{     ... 3]; }': ()
            171..172 'f': &[usize]
            185..195 '&[1, 2, 3]': &[usize; _]
            186..195 '[1, 2, 3]': [usize; _]
            187..188 '1': usize
            190..191 '2': usize
            193..194 '3': usize
        "]],
    );
}

#[test]
fn coerce_unsize_trait_object_simple() {
    check_infer_with_mismatches(
        r#"
        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<T: Unsize<U>, U> CoerceUnsized<&U> for &T {}

        trait Foo<T, U> {}
        trait Bar<U, T, X>: Foo<T, U> {}
        trait Baz<T, X>: Bar<usize, T, X> {}

        struct S<T, X>;
        impl<T, X> Foo<T, usize> for S<T, X> {}
        impl<T, X> Bar<usize, T, X> for S<T, X> {}
        impl<T, X> Baz<T, X> for S<T, X> {}

        fn test() {
            let obj: &dyn Baz<i8, i16> = &S;
            let obj: &dyn Bar<_, i8, i16> = &S;
            let obj: &dyn Foo<i8, _> = &S;
        }
        "#,
        expect![[r"
            424..539 '{     ... &S; }': ()
            434..437 'obj': &dyn Baz<i8, i16>
            459..461 '&S': &S<i8, i16>
            460..461 'S': S<i8, i16>
            471..474 'obj': &dyn Bar<usize, i8, i16>
            499..501 '&S': &S<i8, i16>
            500..501 'S': S<i8, i16>
            511..514 'obj': &dyn Foo<i8, usize>
            534..536 '&S': &S<i8, {unknown}>
            535..536 'S': S<i8, {unknown}>
        "]],
    );
}

#[test]
// The rust reference says this should be possible, but rustc doesn't implement
// it. We used to support it, but Chalk doesn't.
#[ignore]
fn coerce_unsize_trait_object_to_trait_object() {
    check_infer_with_mismatches(
        r#"
        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<T: Unsize<U>, U> CoerceUnsized<&U> for &T {}

        trait Foo<T, U> {}
        trait Bar<U, T, X>: Foo<T, U> {}
        trait Baz<T, X>: Bar<usize, T, X> {}

        struct S<T, X>;
        impl<T, X> Foo<T, usize> for S<T, X> {}
        impl<T, X> Bar<usize, T, X> for S<T, X> {}
        impl<T, X> Baz<T, X> for S<T, X> {}

        fn test() {
            let obj: &dyn Baz<i8, i16> = &S;
            let obj: &dyn Bar<_, _, _> = obj;
            let obj: &dyn Foo<_, _> = obj;
            let obj2: &dyn Baz<i8, i16> = &S;
            let _: &dyn Foo<_, _> = obj2;
        }
        "#,
        expect![[r"
            424..609 '{     ...bj2; }': ()
            434..437 'obj': &dyn Baz<i8, i16>
            459..461 '&S': &S<i8, i16>
            460..461 'S': S<i8, i16>
            471..474 'obj': &dyn Bar<usize, i8, i16>
            496..499 'obj': &dyn Baz<i8, i16>
            509..512 'obj': &dyn Foo<i8, usize>
            531..534 'obj': &dyn Bar<usize, i8, i16>
            544..548 'obj2': &dyn Baz<i8, i16>
            570..572 '&S': &S<i8, i16>
            571..572 'S': S<i8, i16>
            582..583 '_': &dyn Foo<i8, usize>
            602..606 'obj2': &dyn Baz<i8, i16>
        "]],
    );
}

#[test]
fn coerce_unsize_super_trait_cycle() {
    check_infer_with_mismatches(
        r#"
        #[lang = "sized"]
        pub trait Sized {}
        #[lang = "unsize"]
        pub trait Unsize<T> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<T: Unsize<U>, U> CoerceUnsized<&U> for &T {}

        trait A {}
        trait B: C + A {}
        trait C: B {}
        trait D: C

        struct S;
        impl A for S {}
        impl B for S {}
        impl C for S {}
        impl D for S {}

        fn test() {
            let obj: &dyn D = &S;
            let obj: &dyn A = &S;
        }
        "#,
        expect![[r"
            328..383 '{     ... &S; }': ()
            338..341 'obj': &dyn D
            352..354 '&S': &S
            353..354 'S': S
            364..367 'obj': &dyn A
            378..380 '&S': &S
            379..380 'S': S
        "]],
    );
}

#[ignore]
#[test]
fn coerce_unsize_generic() {
    // FIXME: Implement this
    // https://doc.rust-lang.org/reference/type-coercions.html#unsized-coercions
    check_infer_with_mismatches(
        r#"
        #[lang = "unsize"]
        pub trait Unsize<T> {}
        #[lang = "coerce_unsized"]
        pub trait CoerceUnsized<T> {}

        impl<T: Unsize<U>, U> CoerceUnsized<&U> for &T {}

        struct Foo<T> { t: T };
        struct Bar<T>(Foo<T>);

        fn test() {
            let _: &Foo<[usize]> = &Foo { t: [1, 2, 3] };
            let _: &Bar<[usize]> = &Bar(Foo { t: [1, 2, 3] });
        }
        "#,
        expect![[r"
        "]],
    );
}
