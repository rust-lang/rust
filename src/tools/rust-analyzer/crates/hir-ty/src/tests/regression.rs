use expect_test::expect;

use super::{check_infer, check_no_mismatches, check_types};

#[test]
fn bug_484() {
    check_infer(
        r#"
        fn test() {
            let x = if true {};
        }
        "#,
        expect![[r#"
            10..37 '{     ... {}; }': ()
            20..21 'x': ()
            24..34 'if true {}': ()
            27..31 'true': bool
            32..34 '{}': ()
        "#]],
    );
}

#[test]
fn no_panic_on_field_of_enum() {
    check_infer(
        r#"
        enum X {}

        fn test(x: X) {
            x.some_field;
        }
        "#,
        expect![[r#"
            19..20 'x': X
            25..46 '{     ...eld; }': ()
            31..32 'x': X
            31..43 'x.some_field': {unknown}
        "#]],
    );
}

#[test]
fn bug_585() {
    check_infer(
        r#"
        fn test() {
            X {};
            match x {
                A::B {} => (),
                A::Y() => (),
            }
        }
        "#,
        expect![[r#"
            10..88 '{     ...   } }': ()
            16..20 'X {}': {unknown}
            26..86 'match ...     }': ()
            32..33 'x': {unknown}
            44..51 'A::B {}': {unknown}
            55..57 '()': ()
            67..73 'A::Y()': {unknown}
            77..79 '()': ()
        "#]],
    );
}

#[test]
fn bug_651() {
    check_infer(
        r#"
        fn quux() {
            let y = 92;
            1 + y;
        }
        "#,
        expect![[r#"
            10..40 '{     ...+ y; }': ()
            20..21 'y': i32
            24..26 '92': i32
            32..33 '1': i32
            32..37 '1 + y': i32
            36..37 'y': i32
        "#]],
    );
}

#[test]
fn recursive_vars() {
    check_infer(
        r#"
        fn test() {
            let y = unknown;
            [y, &y];
        }
        "#,
        expect![[r#"
            10..47 '{     ...&y]; }': ()
            20..21 'y': {unknown}
            24..31 'unknown': {unknown}
            37..44 '[y, &y]': [{unknown}; 2]
            38..39 'y': {unknown}
            41..43 '&y': &{unknown}
            42..43 'y': {unknown}
        "#]],
    );
}

#[test]
fn recursive_vars_2() {
    check_infer(
        r#"
        fn test() {
            let x = unknown;
            let y = unknown;
            [(x, y), (&y, &x)];
        }
        "#,
        expect![[r#"
            10..79 '{     ...x)]; }': ()
            20..21 'x': &{unknown}
            24..31 'unknown': &{unknown}
            41..42 'y': {unknown}
            45..52 'unknown': {unknown}
            58..76 '[(x, y..., &x)]': [(&{unknown}, {unknown}); 2]
            59..65 '(x, y)': (&{unknown}, {unknown})
            60..61 'x': &{unknown}
            63..64 'y': {unknown}
            67..75 '(&y, &x)': (&{unknown}, {unknown})
            68..70 '&y': &{unknown}
            69..70 'y': {unknown}
            72..74 '&x': &&{unknown}
            73..74 'x': &{unknown}
        "#]],
    );
}

#[test]
fn array_elements_expected_type() {
    check_no_mismatches(
        r#"
        fn test() {
            let x: [[u32; 2]; 2] = [[1, 2], [3, 4]];
        }
        "#,
    );
}

#[test]
fn infer_std_crash_1() {
    // caused stack overflow, taken from std
    check_infer(
        r#"
        enum Maybe<T> {
            Real(T),
            Fake,
        }

        fn write() {
            match something_unknown {
                Maybe::Real(ref mut something) => (),
            }
        }
        "#,
        expect![[r#"
            53..138 '{     ...   } }': ()
            59..136 'match ...     }': ()
            65..82 'someth...nknown': Maybe<{unknown}>
            93..123 'Maybe:...thing)': Maybe<{unknown}>
            105..122 'ref mu...ething': &mut {unknown}
            127..129 '()': ()
        "#]],
    );
}

#[test]
fn infer_std_crash_2() {
    // caused "equating two type variables, ...", taken from std
    check_infer(
        r#"
        fn test_line_buffer() {
            &[0, b'\n', 1, b'\n'];
        }
        "#,
        expect![[r#"
            22..52 '{     ...n']; }': ()
            28..49 '&[0, b...b'\n']': &[u8; 4]
            29..49 '[0, b'...b'\n']': [u8; 4]
            30..31 '0': u8
            33..38 'b'\n'': u8
            40..41 '1': u8
            43..48 'b'\n'': u8
        "#]],
    );
}

#[test]
fn infer_std_crash_3() {
    // taken from rustc
    check_infer(
        r#"
        pub fn compute() {
            match nope!() {
                SizeSkeleton::Pointer { non_zero: true, tail } => {}
            }
        }
        "#,
        expect![[r#"
            17..107 '{     ...   } }': ()
            23..105 'match ...     }': ()
            29..36 'nope!()': {unknown}
            47..93 'SizeSk...tail }': {unknown}
            81..85 'true': bool
            81..85 'true': bool
            87..91 'tail': {unknown}
            97..99 '{}': ()
        "#]],
    );
}

#[test]
fn infer_std_crash_4() {
    // taken from rustc
    check_infer(
        r#"
        pub fn primitive_type() {
            match *self {
                BorrowedRef { type_: Primitive(p), ..} => {},
            }
        }
        "#,
        expect![[r#"
            24..105 '{     ...   } }': ()
            30..103 'match ...     }': ()
            36..41 '*self': {unknown}
            37..41 'self': {unknown}
            52..90 'Borrow...), ..}': {unknown}
            73..85 'Primitive(p)': {unknown}
            83..84 'p': {unknown}
            94..96 '{}': ()
        "#]],
    );
}

#[test]
fn infer_std_crash_5() {
    // taken from rustc
    check_infer(
        r#"
        //- minicore: iterator
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
        "#,
        expect![[r#"
            26..322 '{     ...   } }': ()
            32..320 'for co...     }': fn into_iter<{unknown}>({unknown}) -> <{unknown} as IntoIterator>::IntoIter
            32..320 'for co...     }': {unknown}
            32..320 'for co...     }': !
            32..320 'for co...     }': {unknown}
            32..320 'for co...     }': &mut {unknown}
            32..320 'for co...     }': fn next<{unknown}>(&mut {unknown}) -> Option<<{unknown} as Iterator>::Item>
            32..320 'for co...     }': Option<{unknown}>
            32..320 'for co...     }': ()
            32..320 'for co...     }': ()
            32..320 'for co...     }': ()
            36..43 'content': {unknown}
            47..60 'doesnt_matter': {unknown}
            61..320 '{     ...     }': ()
            75..79 'name': &{unknown}
            82..166 'if doe...     }': &{unknown}
            85..98 'doesnt_matter': bool
            99..128 '{     ...     }': &{unknown}
            113..118 'first': &{unknown}
            134..166 '{     ...     }': &{unknown}
            148..156 '&content': &{unknown}
            149..156 'content': {unknown}
            181..188 'content': &{unknown}
            191..313 'if ICE...     }': &{unknown}
            194..231 'ICE_RE..._VALUE': {unknown}
            194..247 'ICE_RE...&name)': bool
            241..246 '&name': &&{unknown}
            242..246 'name': &{unknown}
            248..276 '{     ...     }': &{unknown}
            262..266 'name': &{unknown}
            282..313 '{     ...     }': {unknown}
            296..303 'content': {unknown}
        "#]],
    );
}

#[test]
fn infer_nested_generics_crash() {
    // another crash found typechecking rustc
    check_infer(
        r#"
        struct Canonical<V> {
            value: V,
        }
        struct QueryResponse<V> {
            value: V,
        }
        fn test<R>(query_response: Canonical<QueryResponse<R>>) {
            &query_response.value;
        }
        "#,
        expect![[r#"
            91..105 'query_response': Canonical<QueryResponse<R>>
            136..166 '{     ...lue; }': ()
            142..163 '&query....value': &QueryResponse<R>
            143..157 'query_response': Canonical<QueryResponse<R>>
            143..163 'query_....value': QueryResponse<R>
        "#]],
    );
}

#[test]
fn infer_paren_macro_call() {
    check_infer(
        r#"
        macro_rules! bar { () => {0u32} }
        fn test() {
            let a = (bar!());
        }
        "#,
        expect![[r#"
            !0..4 '0u32': u32
            44..69 '{     ...()); }': ()
            54..55 'a': u32
        "#]],
    );
}

#[test]
fn infer_array_macro_call() {
    check_infer(
        r#"
        macro_rules! bar { () => {0u32} }
        fn test() {
            let a = [bar!()];
        }
        "#,
        expect![[r#"
            !0..4 '0u32': u32
            44..69 '{     ...()]; }': ()
            54..55 'a': [u32; 1]
            58..66 '[bar!()]': [u32; 1]
        "#]],
    );
}

#[test]
fn bug_1030() {
    check_infer(
        r#"
        struct HashSet<T, H>;
        struct FxHasher;
        type FxHashSet<T> = HashSet<T, FxHasher>;

        impl<T, H> HashSet<T, H> {
            fn default() -> HashSet<T, H> {}
        }

        pub fn main_loop() {
            FxHashSet::default();
        }
        "#,
        expect![[r#"
            143..145 '{}': HashSet<T, H>
            168..197 '{     ...t(); }': ()
            174..192 'FxHash...efault': fn default<{unknown}, FxHasher>() -> HashSet<{unknown}, FxHasher>
            174..194 'FxHash...ault()': HashSet<{unknown}, FxHasher>
        "#]],
    );
}

#[test]
fn issue_2669() {
    check_infer(
        r#"
        trait A {}
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
        }
        "#,
        expect![[r#"
            119..214 '{     ...     }': ()
            129..132 'end': fn end<{unknown}>()
            129..134 'end()': ()
            163..208 '{     ...     }': ()
            181..183 '_x': !
            190..197 'loop {}': !
            195..197 '{}': ()
        "#]],
    )
}

#[test]
fn issue_2705() {
    check_infer(
        r#"
        trait Trait {}
        fn test() {
            <Trait<u32>>::foo()
        }
        "#,
        expect![[r#"
            25..52 '{     ...oo() }': ()
            31..48 '<Trait...>::foo': {unknown}
            31..50 '<Trait...:foo()': ()
        "#]],
    );
}

#[test]
fn issue_2683_chars_impl() {
    check_types(
        r#"
//- minicore: iterator
pub struct Chars<'a> {}
impl<'a> Iterator for Chars<'a> {
    type Item = char;
    fn next(&mut self) -> Option<char> { loop {} }
}

fn test() {
    let chars: Chars<'_>;
    (chars.next(), chars.nth(1));
} //^^^^^^^^^^^^^^^^^^^^^^^^^^^^ (Option<char>, Option<char>)
"#,
    );
}

#[test]
fn issue_3999_slice() {
    check_infer(
        r#"
        fn foo(params: &[usize]) {
            match params {
                [ps @ .., _] => {}
            }
        }
        "#,
        expect![[r#"
            7..13 'params': &[usize]
            25..80 '{     ...   } }': ()
            31..78 'match ...     }': ()
            37..43 'params': &[usize]
            54..66 '[ps @ .., _]': [usize]
            55..62 'ps @ ..': &[usize]
            60..62 '..': [usize]
            64..65 '_': usize
            70..72 '{}': ()
        "#]],
    );
}

#[test]
fn issue_3999_struct() {
    // rust-analyzer should not panic on seeing this malformed
    // record pattern.
    check_infer(
        r#"
        struct Bar {
            a: bool,
        }
        fn foo(b: Bar) {
            match b {
                Bar { a: .. } => {},
            }
        }
        "#,
        expect![[r#"
            35..36 'b': Bar
            43..95 '{     ...   } }': ()
            49..93 'match ...     }': ()
            55..56 'b': Bar
            67..80 'Bar { a: .. }': Bar
            76..78 '..': bool
            84..86 '{}': ()
        "#]],
    );
}

#[test]
fn issue_4235_name_conflicts() {
    check_infer(
        r#"
        struct FOO {}
        static FOO:FOO = FOO {};

        impl FOO {
            fn foo(&self) {}
        }

        fn main() {
            let a = &FOO;
            a.foo();
        }
        "#,
        expect![[r#"
            31..37 'FOO {}': FOO
            63..67 'self': &FOO
            69..71 '{}': ()
            85..119 '{     ...o(); }': ()
            95..96 'a': &FOO
            99..103 '&FOO': &FOO
            100..103 'FOO': FOO
            109..110 'a': &FOO
            109..116 'a.foo()': ()
        "#]],
    );
}

#[test]
fn issue_4465_dollar_crate_at_type() {
    check_infer(
        r#"
        pub struct Foo {}
        pub fn anything<T>() -> T {
            loop {}
        }
        macro_rules! foo {
            () => {{
                let r: $crate::Foo = anything();
                r
            }};
        }
        fn main() {
            let _a = foo!();
        }
        "#,
        expect![[r#"
            44..59 '{     loop {} }': T
            50..57 'loop {}': !
            55..57 '{}': ()
            !0..31 '{letr:...g();r}': Foo
            !4..5 'r': Foo
            !18..26 'anything': fn anything<Foo>() -> Foo
            !18..28 'anything()': Foo
            !29..30 'r': Foo
            163..187 '{     ...!(); }': ()
            173..175 '_a': Foo
        "#]],
    );
}

#[test]
fn issue_6811() {
    check_infer(
        r#"
        macro_rules! profile_function {
            () => {
                let _a = 1;
                let _b = 1;
            };
        }
        fn main() {
            profile_function!();
        }
        "#,
        expect![[r#"
            !3..5 '_a': i32
            !6..7 '1': i32
            !11..13 '_b': i32
            !14..15 '1': i32
            103..131 '{     ...!(); }': ()
        "#]],
    );
}

#[test]
fn issue_4053_diesel_where_clauses() {
    check_infer(
        r#"
        trait BoxedDsl<DB> {
            type Output;
            fn internal_into_boxed(self) -> Self::Output;
        }

        struct SelectStatement<From, Select, Distinct, Where, Order, LimitOffset, GroupBy, Locking> {
            order: Order,
        }

        trait QueryFragment<DB: Backend> {}

        trait Into<T> { fn into(self) -> T; }

        impl<F, S, D, W, O, LOf, DB> BoxedDsl<DB>
            for SelectStatement<F, S, D, W, O, LOf, G>
        where
            O: Into<dyn QueryFragment<DB>>,
        {
            type Output = XXX;

            fn internal_into_boxed(self) -> Self::Output {
                self.order.into();
            }
        }
        "#,
        expect![[r#"
            65..69 'self': Self
            267..271 'self': Self
            466..470 'self': SelectStatement<F, S, D, W, O, LOf, {unknown}, {unknown}>
            488..522 '{     ...     }': ()
            498..502 'self': SelectStatement<F, S, D, W, O, LOf, {unknown}, {unknown}>
            498..508 'self.order': O
            498..515 'self.o...into()': dyn QueryFragment<DB>
        "#]],
    );
}

#[test]
fn issue_4953() {
    check_infer(
        r#"
        pub struct Foo(pub i64);
        impl Foo {
            fn test() -> Self { Self(0i64) }
        }
        "#,
        expect![[r#"
            58..72 '{ Self(0i64) }': Foo
            60..64 'Self': Foo(i64) -> Foo
            60..70 'Self(0i64)': Foo
            65..69 '0i64': i64
        "#]],
    );
    check_infer(
        r#"
        pub struct Foo<T>(pub T);
        impl Foo<i64> {
            fn test() -> Self { Self(0i64) }
        }
        "#,
        expect![[r#"
            64..78 '{ Self(0i64) }': Foo<i64>
            66..70 'Self': Foo<i64>(i64) -> Foo<i64>
            66..76 'Self(0i64)': Foo<i64>
            71..75 '0i64': i64
        "#]],
    );
}

#[test]
fn issue_4931() {
    check_infer(
        r#"
        trait Div<T> {
            type Output;
        }

        trait CheckedDiv: Div<()> {}

        trait PrimInt: CheckedDiv<Output = ()> {
            fn pow(self);
        }

        fn check<T: PrimInt>(i: T) {
            i.pow();
        }
        "#,
        expect![[r#"
            117..121 'self': Self
            148..149 'i': T
            154..170 '{     ...w(); }': ()
            160..161 'i': T
            160..167 'i.pow()': ()
        "#]],
    );
}

#[test]
fn issue_4885() {
    check_infer(
        r#"
        //- minicore: coerce_unsized, future
        use core::future::Future;
        trait Foo<R> {
            type Bar;
        }
        fn foo<R, K>(key: &K) -> impl Future<Output = K::Bar>
        where
            K: Foo<R>,
        {
            bar(key)
        }
        fn bar<R, K>(key: &K) -> impl Future<Output = K::Bar>
        where
            K: Foo<R>,
        {
        }
        "#,
        expect![[r#"
            70..73 'key': &K
            132..148 '{     ...key) }': impl Future<Output = <K as Foo<R>>::Bar>
            138..141 'bar': fn bar<R, K>(&K) -> impl Future<Output = <K as Foo<R>>::Bar>
            138..146 'bar(key)': impl Future<Output = <K as Foo<R>>::Bar>
            142..145 'key': &K
            162..165 'key': &K
            224..227 '{ }': ()
        "#]],
    );
}

#[test]
fn issue_4800() {
    check_infer(
        r#"
        trait Debug {}

        struct Foo<T>;

        type E1<T> = (T, T, T);
        type E2<T> = E1<E1<E1<(T, T, T)>>>;

        impl Debug for Foo<E2<()>> {}

        struct Request;

        pub trait Future {
            type Output;
        }

        pub struct PeerSet<D>;

        impl<D> Service<Request> for PeerSet<D>
        where
            D: Discover,
            D::Key: Debug,
        {
            type Error = ();
            type Future = dyn Future<Output = Self::Error>;

            fn call(&mut self) -> Self::Future {
                loop {}
            }
        }

        pub trait Discover {
            type Key;
        }

        pub trait Service<Request> {
            type Error;
            type Future: Future<Output = Self::Error>;
            fn call(&mut self) -> Self::Future;
        }
        "#,
        expect![[r#"
            379..383 'self': &mut PeerSet<D>
            401..424 '{     ...     }': dyn Future<Output = ()>
            411..418 'loop {}': !
            416..418 '{}': ()
            575..579 'self': &mut Self
        "#]],
    );
}

#[test]
fn issue_4966() {
    check_infer(
        r#"
        //- minicore: deref
        pub trait IntoIterator {
            type Item;
        }

        struct Repeat<A> { element: A }

        struct Map<F> { f: F }

        struct Vec<T> {}

        impl<T> core::ops::Deref for Vec<T> {
            type Target = [T];
        }

        fn from_iter<A, T: IntoIterator<Item = A>>(iter: T) -> Vec<A> {}

        fn main() {
            let inner = Map { f: |_: &f64| 0.0 };

            let repeat = Repeat { element: inner };

            let vec = from_iter(repeat);

            vec.foo_bar();
        }
        "#,
        expect![[r#"
            225..229 'iter': T
            244..246 '{}': Vec<A>
            258..402 '{     ...r(); }': ()
            268..273 'inner': Map<impl Fn(&f64) -> f64>
            276..300 'Map { ... 0.0 }': Map<impl Fn(&f64) -> f64>
            285..298 '|_: &f64| 0.0': impl Fn(&f64) -> f64
            286..287 '_': &f64
            295..298 '0.0': f64
            311..317 'repeat': Repeat<Map<impl Fn(&f64) -> f64>>
            320..345 'Repeat...nner }': Repeat<Map<impl Fn(&f64) -> f64>>
            338..343 'inner': Map<impl Fn(&f64) -> f64>
            356..359 'vec': Vec<IntoIterator::Item<Repeat<Map<impl Fn(&f64) -> f64>>>>
            362..371 'from_iter': fn from_iter<IntoIterator::Item<Repeat<Map<impl Fn(&f64) -> f64>>>, Repeat<Map<impl Fn(&f64) -> f64>>>(Repeat<Map<impl Fn(&f64) -> f64>>) -> Vec<IntoIterator::Item<Repeat<Map<impl Fn(&f64) -> f64>>>>
            362..379 'from_i...epeat)': Vec<IntoIterator::Item<Repeat<Map<impl Fn(&f64) -> f64>>>>
            372..378 'repeat': Repeat<Map<impl Fn(&f64) -> f64>>
            386..389 'vec': Vec<IntoIterator::Item<Repeat<Map<impl Fn(&f64) -> f64>>>>
            386..399 'vec.foo_bar()': {unknown}
        "#]],
    );
}

#[test]
fn issue_6628() {
    check_infer(
        r#"
//- minicore: fn
struct S<T>();
impl<T> S<T> {
    fn f(&self, _t: T) {}
    fn g<F: FnOnce(&T)>(&self, _f: F) {}
}
fn main() {
    let s = S();
    s.g(|_x| {});
    s.f(10);
}
"#,
        expect![[r#"
            40..44 'self': &S<T>
            46..48 '_t': T
            53..55 '{}': ()
            81..85 'self': &S<T>
            87..89 '_f': F
            94..96 '{}': ()
            109..160 '{     ...10); }': ()
            119..120 's': S<i32>
            123..124 'S': S<i32>() -> S<i32>
            123..126 'S()': S<i32>
            132..133 's': S<i32>
            132..144 's.g(|_x| {})': ()
            136..143 '|_x| {}': impl Fn(&i32)
            137..139 '_x': &i32
            141..143 '{}': ()
            150..151 's': S<i32>
            150..157 's.f(10)': ()
            154..156 '10': i32
        "#]],
    );
}

#[test]
fn issue_6852() {
    check_infer(
        r#"
//- minicore: deref
use core::ops::Deref;

struct BufWriter {}

struct Mutex<T> {}
struct MutexGuard<'a, T> {}
impl<T> Mutex<T> {
    fn lock(&self) -> MutexGuard<'_, T> {}
}
impl<'a, T: 'a> Deref for MutexGuard<'a, T> {
    type Target = T;
}
fn flush(&self) {
    let w: &Mutex<BufWriter>;
    *(w.lock());
}
"#,
        expect![[r#"
            123..127 'self': &Mutex<T>
            150..152 '{}': MutexGuard<'_, T>
            234..238 'self': &{unknown}
            240..290 '{     ...()); }': ()
            250..251 'w': &Mutex<BufWriter>
            276..287 '*(w.lock())': BufWriter
            278..279 'w': &Mutex<BufWriter>
            278..286 'w.lock()': MutexGuard<'_, BufWriter>
        "#]],
    );
}

#[test]
fn param_overrides_fn() {
    check_types(
        r#"
        fn example(example: i32) {
            fn f() {}
            example;
          //^^^^^^^ i32
        }
        "#,
    )
}

#[test]
fn lifetime_from_chalk_during_deref() {
    check_types(
        r#"
//- minicore: deref
struct Box<T: ?Sized> {}
impl<T: ?Sized> core::ops::Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        loop {}
    }
}

trait Iterator {
    type Item;
}

pub struct Iter<'a, T: 'a> {
    inner: Box<dyn IterTrait<'a, T, Item = &'a T> + 'a>,
}

trait IterTrait<'a, T: 'a>: Iterator<Item = &'a T> {
    fn clone_box(&self);
}

fn clone_iter<T>(s: Iter<T>) {
    s.inner.clone_box();
  //^^^^^^^^^^^^^^^^^^^ ()
}
"#,
    )
}

#[test]
fn issue_8686() {
    check_infer(
        r#"
pub trait Try: FromResidual {
    type Output;
    type Residual;
}
pub trait FromResidual<R = <Self as Try>::Residual> {
     fn from_residual(residual: R) -> Self;
}

struct ControlFlow<B, C>;
impl<B, C> Try for ControlFlow<B, C> {
    type Output = C;
    type Residual = ControlFlow<B, !>;
}
impl<B, C> FromResidual for ControlFlow<B, C> {
    fn from_residual(r: ControlFlow<B, !>) -> Self { ControlFlow }
}

fn test() {
    ControlFlow::from_residual(ControlFlow::<u32, !>);
}
        "#,
        expect![[r#"
            144..152 'residual': R
            365..366 'r': ControlFlow<B, !>
            395..410 '{ ControlFlow }': ControlFlow<B, C>
            397..408 'ControlFlow': ControlFlow<B, C>
            424..482 '{     ...!>); }': ()
            430..456 'Contro...sidual': fn from_residual<ControlFlow<u32, {unknown}>, ControlFlow<u32, !>>(ControlFlow<u32, !>) -> ControlFlow<u32, {unknown}>
            430..479 'Contro...2, !>)': ControlFlow<u32, {unknown}>
            457..478 'Contro...32, !>': ControlFlow<u32, !>
        "#]],
    );
}

#[test]
fn cfg_tail() {
    // https://github.com/rust-lang/rust-analyzer/issues/8378
    check_infer(
        r#"
        fn fake_tail(){
            { "first" }
            #[cfg(never)] 9
        }
        fn multiple_fake(){
            { "fake" }
            { "fake" }
            { "second" }
            #[cfg(never)] { 11 }
            #[cfg(never)] 12;
            #[cfg(never)] 13
        }
        fn no_normal_tail(){
            { "third" }
            #[cfg(never)] 14;
            #[cfg(never)] 15;
        }
        fn no_actual_tail(){
            { "fourth" };
            #[cfg(never)] 14;
            #[cfg(never)] 15
        }
        "#,
        expect![[r#"
            14..53 '{     ...)] 9 }': ()
            20..31 '{ "first" }': ()
            22..29 '"first"': &str
            72..190 '{     ...] 13 }': ()
            78..88 '{ "fake" }': ()
            80..86 '"fake"': &str
            93..103 '{ "fake" }': ()
            95..101 '"fake"': &str
            108..120 '{ "second" }': ()
            110..118 '"second"': &str
            210..273 '{     ... 15; }': ()
            216..227 '{ "third" }': ()
            218..225 '"third"': &str
            293..357 '{     ...] 15 }': ()
            299..311 '{ "fourth" }': &str
            301..309 '"fourth"': &str
        "#]],
    )
}

#[test]
fn impl_trait_in_option_9530() {
    check_types(
        r#"
//- minicore: sized
struct Option<T>;
impl<T> Option<T> {
    fn unwrap(self) -> T { loop {} }
}
fn make() -> Option<impl Copy> { Option }
trait Copy {}
fn test() {
    let o = make();
    o.unwrap();
  //^^^^^^^^^^ impl Copy
}
        "#,
    )
}

#[test]
fn bare_dyn_trait_binders_9639() {
    check_no_mismatches(
        r#"
//- minicore: fn, coerce_unsized
fn infix_parse<T, S>(_state: S, _level_code: &Fn(S)) -> T {
    loop {}
}

fn parse_a_rule() {
    infix_parse((), &(|_recurse| ()))
}
        "#,
    )
}

#[test]
fn nested_closure() {
    check_types(
        r#"
//- minicore: fn, option

fn map<T, U>(o: Option<T>, f: impl FnOnce(T) -> U) -> Option<U> { loop {} }

fn test() {
    let o = Some(Some(2));
    map(o, |s| map(s, |x| x));
                    // ^ i32
}
        "#,
    );
}

#[test]
fn call_expected_type_closure() {
    check_types(
        r#"
//- minicore: fn, option

fn map<T, U>(o: Option<T>, f: impl FnOnce(T) -> U) -> Option<U> { loop {} }
struct S {
    field: u32
}

fn test() {
    let o = Some(S { field: 2 });
    let _: Option<()> = map(o, |s| { s.field; });
                                  // ^^^^^^^ u32
}
        "#,
    );
}

#[test]
fn coerce_diesel_panic() {
    check_no_mismatches(
        r#"
//- minicore: option

trait TypeMetadata {
    type MetadataLookup;
}

pub struct Output<'a, T, DB>
where
    DB: TypeMetadata,
    DB::MetadataLookup: 'a,
{
    out: T,
    metadata_lookup: Option<&'a DB::MetadataLookup>,
}

impl<'a, T, DB: TypeMetadata> Output<'a, T, DB> {
    pub fn new(out: T, metadata_lookup: &'a DB::MetadataLookup) -> Self {
        Output {
            out,
            metadata_lookup: Some(metadata_lookup),
        }
    }
}
        "#,
    );
}

#[test]
fn bitslice_panic() {
    check_no_mismatches(
        r#"
//- minicore: option, deref

pub trait BitView {
    type Store;
}

pub struct Lsb0;

pub struct BitArray<V: BitView> { }

pub struct BitSlice<T> { }

impl<V: BitView> core::ops::Deref for BitArray<V> {
    type Target = BitSlice<V::Store>;
}

impl<T> BitSlice<T> {
    pub fn split_first(&self) -> Option<(T, &Self)> { loop {} }
}

fn multiexp_inner() {
    let exp: &BitArray<Foo>;
    exp.split_first();
}
        "#,
    );
}

#[test]
fn macro_expands_to_impl_trait() {
    check_no_mismatches(
        r#"
trait Foo {}

macro_rules! ty {
    () => {
        impl Foo
    }
}

fn foo(_: ty!()) {}

fn bar() {
    foo(());
}
    "#,
    )
}

#[test]
fn nested_macro_in_fn_params() {
    check_no_mismatches(
        r#"
macro_rules! U32Inner {
    () => {
        u32
    };
}

macro_rules! U32 {
    () => {
        U32Inner!()
    };
}

fn mamba(a: U32!(), p: u32) -> u32 {
    a
}
    "#,
    )
}

#[test]
fn for_loop_block_expr_iterable() {
    check_infer(
        r#"
//- minicore: iterator
fn test() {
    for _ in { let x = 0; } {
        let y = 0;
    }
}
        "#,
        expect![[r#"
            10..68 '{     ...   } }': ()
            16..66 'for _ ...     }': fn into_iter<()>(()) -> <() as IntoIterator>::IntoIter
            16..66 'for _ ...     }': IntoIterator::IntoIter<()>
            16..66 'for _ ...     }': !
            16..66 'for _ ...     }': IntoIterator::IntoIter<()>
            16..66 'for _ ...     }': &mut IntoIterator::IntoIter<()>
            16..66 'for _ ...     }': fn next<IntoIterator::IntoIter<()>>(&mut IntoIterator::IntoIter<()>) -> Option<<IntoIterator::IntoIter<()> as Iterator>::Item>
            16..66 'for _ ...     }': Option<Iterator::Item<IntoIterator::IntoIter<()>>>
            16..66 'for _ ...     }': ()
            16..66 'for _ ...     }': ()
            16..66 'for _ ...     }': ()
            20..21 '_': Iterator::Item<IntoIterator::IntoIter<()>>
            25..39 '{ let x = 0; }': ()
            31..32 'x': i32
            35..36 '0': i32
            40..66 '{     ...     }': ()
            54..55 'y': i32
            58..59 '0': i32
        "#]],
    );
}

#[test]
fn while_loop_block_expr_iterable() {
    check_infer(
        r#"
fn test() {
    while { true } {
        let y = 0;
    }
}
        "#,
        expect![[r#"
            10..59 '{     ...   } }': ()
            16..57 'while ...     }': ()
            22..30 '{ true }': bool
            24..28 'true': bool
            31..57 '{     ...     }': ()
            45..46 'y': i32
            49..50 '0': i32
        "#]],
    );
}

#[test]
fn bug_11242() {
    // FIXME: wrong, should be u32
    check_types(
        r#"
fn foo<A, B>()
where
    A: IntoIterator<Item = u32>,
    B: IntoIterator<Item = usize>,
{
    let _x: <A as IntoIterator>::Item;
     // ^^ {unknown}
}

pub trait Iterator {
    type Item;
}

pub trait IntoIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
}

impl<I: Iterator> IntoIterator for I {
    type Item = I::Item;
    type IntoIter = I;
}
"#,
    );
}

#[test]
fn bug_11659() {
    check_no_mismatches(
        r#"
struct LinkArray<const N: usize, LD>(LD);
fn f<const N: usize, LD>(x: LD) -> LinkArray<N, LD> {
    let r = LinkArray::<N, LD>(x);
    r
}

fn test() {
    let x = f::<2, i32>(5);
    let y = LinkArray::<52, LinkArray<2, i32>>(x);
}
        "#,
    );
    check_no_mismatches(
        r#"
struct LinkArray<LD, const N: usize>(LD);
fn f<const N: usize, LD>(x: LD) -> LinkArray<LD, N> {
    let r = LinkArray::<LD, N>(x);
    r
}

fn test() {
    let x = f::<i32, 2>(5);
    let y = LinkArray::<LinkArray<i32, 2>, 52>(x);
}
        "#,
    );
}

#[test]
fn const_generic_error_tolerance() {
    check_no_mismatches(
        r#"
#[lang = "sized"]
pub trait Sized {}

struct CT<const N: usize, T>(T);
struct TC<T, const N: usize>(T);
fn f<const N: usize, T>(x: T) -> (CT<N, T>, TC<T, N>) {
    let l = CT::<N, T>(x);
    let r = TC::<N, T>(x);
    (l, r)
}

trait TR1<const N: usize>;
trait TR2<const N: usize>;

impl<const N: usize, T> TR1<N> for CT<N, T>;
impl<const N: usize, T> TR1<5> for TC<T, N>;
impl<const N: usize, T> TR2<N> for CT<T, N>;

trait TR3<const N: usize> {
    fn tr3(&self) -> &Self;
}

impl<const N: usize, T> TR3<5> for TC<T, N> {
    fn tr3(&self) -> &Self {
        self
    }
}

impl<const N: usize, T> TR3<Item = 5> for TC<T, N> {}
impl<const N: usize, T> TR3<T> for TC<T, N> {}

fn impl_trait<const N: usize>(inp: impl TR1<N>) {}
fn dyn_trait<const N: usize>(inp: &dyn TR2<N>) {}
fn impl_trait_bad<'a, const N: usize>(inp: impl TR1<i32>) -> impl TR1<'a, i32> {}
fn impl_trait_very_bad<const N: usize>(inp: impl TR1<Item = i32>) -> impl TR1<'a, Item = i32, 5, Foo = N> {}

fn test() {
    f::<2, i32>(5);
    f::<2, 2>(5);
    f(5);
    f::<i32>(5);
    CT::<52, CT<2, i32>>(x);
    CT::<CT<2, i32>>(x);
    impl_trait_bad(5);
    impl_trait_bad(12);
    TR3<5>::tr3();
    TR3<{ 2+3 }>::tr3();
    TC::<i32, 10>(5).tr3();
    TC::<i32, 20>(5).tr3();
    TC::<i32, i32>(5).tr3();
    TC::<i32, { 7 + 3 }>(5).tr3();
}
        "#,
    );
}

#[test]
fn const_generic_impl_trait() {
    check_no_mismatches(
        r#"
        //- minicore: from

        struct Foo<T, const M: usize>;

        trait Tr<T> {
            fn f(T) -> Self;
        }

        impl<T, const M: usize> Tr<[T; M]> for Foo<T, M> {
            fn f(_: [T; M]) -> Self {
                Self
            }
        }

        fn test() {
            Foo::f([1, 2, 7, 10]);
        }
        "#,
    );
}

#[test]
fn nalgebra_factorial() {
    check_no_mismatches(
        r#"
        const FACTORIAL: [u128; 4] = [1, 1, 2, 6];

        fn factorial(n: usize) -> u128 {
            match FACTORIAL.get(n) {
                Some(f) => *f,
                None => panic!("{}! is greater than u128::MAX", n),
            }
        }
        "#,
    )
}

#[test]
fn regression_11688_1() {
    check_no_mismatches(
        r#"
        pub struct Buffer<T>(T);
        type Writer = Buffer<u8>;
        impl<T> Buffer<T> {
            fn extend_from_array<const N: usize>(&mut self, xs: &[T; N]) {
                loop {}
            }
        }
        trait Encode<S> {
            fn encode(self, w: &mut Writer, s: &mut S);
        }
        impl<S> Encode<S> for u8 {
            fn encode(self, w: &mut Writer, _: &mut S) {
                w.extend_from_array(&self.to_le_bytes());
            }
        }
        "#,
    );
}

#[test]
fn regression_11688_2() {
    check_types(
        r#"
        union MaybeUninit<T> {
            uninit: (),
            value: T,
        }

        impl<T> MaybeUninit<T> {
            fn uninit_array<const LEN: usize>() -> [Self; LEN] {
                loop {}
            }
        }

        fn main() {
            let x = MaybeUninit::<i32>::uninit_array::<1>();
              //^ [MaybeUninit<i32>; 1]
        }
        "#,
    );
}

#[test]
fn regression_11688_3() {
    check_types(
        r#"
        //- minicore: iterator
        struct Ar<T, const N: u8>(T);
        fn f<const LEN: usize, T, const BASE: u8>(
            num_zeros: usize,
        ) -> &dyn Iterator<Item = [Ar<T, BASE>; LEN]> {
            loop {}
        }
        fn dynamic_programming() {
            let board = f::<9, u8, 7>(1).next();
              //^^^^^ Option<[Ar<u8, 7>; 9]>
        }
        "#,
    );
}

#[test]
fn regression_11688_4() {
    check_types(
        r#"
        trait Bar<const C: usize> {
            fn baz(&self) -> [i32; C];
        }

        fn foo(x: &dyn Bar<2>) {
            x.baz();
          //^^^^^^^ [i32; 2]
        }
        "#,
    )
}

#[test]
fn gat_crash_1() {
    check_no_mismatches(
        r#"
trait ATrait {}

trait Crash {
    type Member<const N: usize>: ATrait;
    fn new<const N: usize>() -> Self::Member<N>;
}

fn test<T: Crash>() {
    T::new();
}
"#,
    );
}

#[test]
fn gat_crash_2() {
    check_no_mismatches(
        r#"
pub struct InlineStorage {}

pub struct InlineStorageHandle<T: ?Sized> {}

pub unsafe trait Storage {
    type Handle<T: ?Sized>;
    fn create<T: ?Sized>() -> Self::Handle<T>;
}

unsafe impl Storage for InlineStorage {
    type Handle<T: ?Sized> = InlineStorageHandle<T>;
}
"#,
    );
}

#[test]
fn gat_crash_3() {
    check_no_mismatches(
        r#"
trait Collection {
type Item;
type Member<T>: Collection<Item = T>;
fn add(&mut self, value: Self::Item) -> Result<(), Self::Error>;
}
struct ConstGen<T, const N: usize> {
data: [T; N],
}
impl<T, const N: usize> Collection for ConstGen<T, N> {
type Item = T;
type Member<U> = ConstGen<U, N>;
}
    "#,
    );
}

#[test]
fn cfgd_out_self_param() {
    cov_mark::check!(cfgd_out_self_param);
    check_no_mismatches(
        r#"
struct S;
impl S {
    fn f(#[cfg(never)] &self) {}
}

fn f(s: S) {
    s.f();
}
"#,
    );
}

#[test]
fn rust_161_option_clone() {
    check_types(
        r#"
//- minicore: option, drop

fn test(o: &Option<i32>) {
    o.my_clone();
  //^^^^^^^^^^^^ Option<i32>
}

pub trait MyClone: Sized {
    fn my_clone(&self) -> Self;
}

impl<T> const MyClone for Option<T>
where
    T: ~const MyClone + ~const Drop + ~const Destruct,
{
    fn my_clone(&self) -> Self {
        match self {
            Some(x) => Some(x.my_clone()),
            None => None,
        }
    }
}

impl const MyClone for i32 {
    fn my_clone(&self) -> Self {
        *self
    }
}

pub trait Destruct {}

impl<T: ?Sized> const Destruct for T {}
"#,
    );
}

#[test]
fn rust_162_option_clone() {
    check_types(
        r#"
//- minicore: option, drop

fn test(o: &Option<i32>) {
    o.my_clone();
  //^^^^^^^^^^^^ Option<i32>
}

pub trait MyClone: Sized {
    fn my_clone(&self) -> Self;
}

impl<T> const MyClone for Option<T>
where
    T: ~const MyClone + ~const Destruct,
{
    fn my_clone(&self) -> Self {
        match self {
            Some(x) => Some(x.my_clone()),
            None => None,
        }
    }
}

impl const MyClone for i32 {
    fn my_clone(&self) -> Self {
        *self
    }
}

#[lang = "destruct"]
pub trait Destruct {}
"#,
    );
}

#[test]
fn tuple_struct_pattern_with_unmatched_args_crash() {
    check_infer(
        r#"
struct S(usize);
fn main() {
    let S(.., a, b) = S(1);
    let (.., a, b) = (1,);
}
        "#,
        expect![[r#"
        27..85 '{     ...1,); }': ()
        37..48 'S(.., a, b)': S
        43..44 'a': usize
        46..47 'b': {unknown}
        51..52 'S': S(usize) -> S
        51..55 'S(1)': S
        53..54 '1': usize
        65..75 '(.., a, b)': (i32, {unknown})
        70..71 'a': i32
        73..74 'b': {unknown}
        78..82 '(1,)': (i32,)
        79..80 '1': i32
        "#]],
    );
}

#[test]
fn trailing_empty_macro() {
    check_no_mismatches(
        r#"
macro_rules! m2 {
    ($($t:tt)*) => {$($t)*};
}

fn macrostmts() -> u8 {
    m2! { 0 }
    m2! {}
}
    "#,
    );
}

#[test]
fn dyn_with_unresolved_trait() {
    check_types(
        r#"
fn foo(a: &dyn DoesNotExist) {
    a.bar();
  //^&{unknown}
}
        "#,
    );
}

#[test]
fn self_assoc_with_const_generics_crash() {
    check_no_mismatches(
        r#"
trait Trait { type Item; }
impl<T, const N: usize> Trait for [T; N] {
    type Item = ();
    fn f<U>(_: Self::Item) {}
}
        "#,
    );
}

#[test]
fn unsize_array_with_inference_variable() {
    check_types(
        r#"
//- minicore: try, slice
use core::ops::ControlFlow;
fn foo() -> ControlFlow<(), [usize; 1]> { loop {} }
fn bar() -> ControlFlow<(), ()> {
    let a = foo()?.len();
      //^ usize
    ControlFlow::Continue(())
}
"#,
    );
}

#[test]
fn assoc_type_shorthand_with_gats_in_binders() {
    // c.f. test `issue_4885()`
    check_no_mismatches(
        r#"
trait Gats {
    type Assoc<T>;
}
trait Foo<T> {}

struct Bar<'a, B: Gats, A> {
    field: &'a dyn Foo<B::Assoc<A>>,
}

fn foo(b: Bar) {
    let _ = b.field;
}
"#,
    );
}

#[test]
fn regression_14305() {
    check_no_mismatches(
        r#"
//- minicore: add
trait Tr {}
impl Tr for [u8; C] {}
const C: usize = 2 + 2;
"#,
    );
}

#[test]
fn regression_14456() {
    check_types(
        r#"
//- minicore: future
async fn x() {}
fn f() {
    let fut = x();
    let t = [0u8; { let a = 2 + 2; a }];
      //^ [u8; 4]
}
"#,
    );
}

#[test]
fn regression_14164() {
    check_types(
        r#"
trait Rec {
    type K;
    type Rebind<Tok>: Rec<K = Tok>;
}

trait Expr<K> {
    type Part: Rec<K = K>;
    fn foo(_: <Self::Part as Rec>::Rebind<i32>) {}
}

struct Head<K>(K);
impl<K> Rec for Head<K> {
    type K = K;
    type Rebind<Tok> = Head<Tok>;
}

fn test<E>()
where
    E: Expr<usize, Part = Head<usize>>,
{
    let head;
      //^^^^ Head<i32>
    E::foo(head);
}
"#,
    );
}

#[test]
fn match_ergonomics_with_binding_modes_interaction() {
    check_types(
        r"
enum E { A }
fn foo() {
    match &E::A {
        b @ (x @ E::A | x) => {
            b;
          //^ &E
            x;
          //^ &E
        }
    }
}",
    );
}

#[test]
fn regression_14844() {
    check_no_mismatches(
        r#"
pub type Ty = Unknown;

pub struct Inner<T>();

pub struct Outer {
    pub inner: Inner<Ty>,
}

fn main() {
    _ = Outer {
        inner: Inner::<i32>(),
    };
}
        "#,
    );
    check_no_mismatches(
        r#"
pub const ONE: usize = 1;

pub struct Inner<const P: usize>();

pub struct Outer {
    pub inner: Inner<ONE>,
}

fn main() {
    _ = Outer {
        inner: Inner::<1>(),
    };
}
        "#,
    );
    check_no_mismatches(
        r#"
pub const ONE: usize = unknown();

pub struct Inner<const P: usize>();

pub struct Outer {
    pub inner: Inner<ONE>,
}

fn main() {
    _ = Outer {
        inner: Inner::<1>(),
    };
}
        "#,
    );
    check_no_mismatches(
        r#"
pub const N: usize = 2 + 2;

fn f(t: [u8; N]) {}

fn main() {
    let a = [1, 2, 3, 4];
    f(a);
    let b = [1; 4];
    let c: [u8; N] = b;
    let d = [1; N];
    let e: [u8; N] = d;
    let f = [1; N];
    let g = match f {
        [a, b, c, d] => a + b + c + d,
    };
}
        "#,
    );
}

#[test]
fn regression_14844_2() {
    check_no_mismatches(
        r#"
//- minicore: fn
pub const ONE: usize = 1;

pub type MyInner = Inner<ONE>;

pub struct Inner<const P: usize>();

impl Inner<1> {
    fn map<F>(&self, func: F) -> bool
    where
        F: Fn(&MyInner) -> bool,
    {
        func(self)
    }
}
        "#,
    );
}

#[test]
fn dont_crash_on_slice_unsizing() {
    check_no_mismatches(
        r#"
//- minicore: slice, unsize, coerce_unsized
trait Tr {
    fn f(self);
}

impl Tr for [i32] {
    fn f(self) {
        let t;
        x(t);
    }
}

fn x(a: [i32; 4]) {
    let b = a.f();
}
        "#,
    );
}
