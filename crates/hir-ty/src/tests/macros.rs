use expect_test::expect;
use test_utils::{bench, bench_fixture, skip_slow_tests};

use crate::tests::check_infer_with_mismatches;

use super::{check_infer, check_types};

#[test]
fn cfg_impl_def() {
    check_types(
        r#"
//- /main.rs crate:main deps:foo cfg:test
use foo::S as T;
struct S;

#[cfg(test)]
impl S {
    fn foo1(&self) -> i32 { 0 }
}

#[cfg(not(test))]
impl S {
    fn foo2(&self) -> i32 { 0 }
}

fn test() {
    let t = (S.foo1(), S.foo2(), T.foo3(), T.foo4());
    t;
} //^ (i32, {unknown}, i32, {unknown})

//- /foo.rs crate:foo
pub struct S;

#[cfg(not(test))]
impl S {
    pub fn foo3(&self) -> i32 { 0 }
}

#[cfg(test)]
impl S {
    pub fn foo4(&self) -> i32 { 0 }
}
"#,
    );
}

#[test]
fn infer_macros_expanded() {
    check_infer(
        r#"
        struct Foo(Vec<i32>);

        macro_rules! foo {
            ($($item:expr),*) => {
                    {
                        Foo(vec![$($item,)*])
                    }
            };
        }

        fn main() {
            let x = foo!(1,2);
        }
        "#,
        expect![[r#"
            !0..17 '{Foo(v...,2,])}': Foo
            !1..4 'Foo': Foo({unknown}) -> Foo
            !1..16 'Foo(vec![1,2,])': Foo
            !5..15 'vec![1,2,]': {unknown}
            155..181 '{     ...,2); }': ()
            165..166 'x': Foo
        "#]],
    );
}

#[test]
fn infer_legacy_textual_scoped_macros_expanded() {
    check_infer(
        r#"
        struct Foo(Vec<i32>);

        #[macro_use]
        mod m {
            macro_rules! foo {
                ($($item:expr),*) => {
                    {
                        Foo(vec![$($item,)*])
                    }
                };
            }
        }

        fn main() {
            let x = foo!(1,2);
            let y = crate::foo!(1,2);
        }
        "#,
        expect![[r#"
            !0..17 '{Foo(v...,2,])}': Foo
            !1..4 'Foo': Foo({unknown}) -> Foo
            !1..16 'Foo(vec![1,2,])': Foo
            !5..15 'vec![1,2,]': {unknown}
            194..250 '{     ...,2); }': ()
            204..205 'x': Foo
            227..228 'y': {unknown}
            231..247 'crate:...!(1,2)': {unknown}
        "#]],
    );
}

#[test]
fn infer_path_qualified_macros_expanded() {
    check_infer(
        r#"
        #[macro_export]
        macro_rules! foo {
            () => { 42i32 }
        }

        mod m {
            pub use super::foo as bar;
        }

        fn main() {
            let x = crate::foo!();
            let y = m::bar!();
        }
        "#,
        expect![[r#"
            !0..5 '42i32': i32
            !0..5 '42i32': i32
            110..163 '{     ...!(); }': ()
            120..121 'x': i32
            147..148 'y': i32
        "#]],
    );
}

#[test]
fn expr_macro_def_expanded_in_various_places() {
    check_infer(
        r#"
        //- minicore: iterator
        macro spam() {
            1isize
        }

        fn spam() {
            spam!();
            (spam!());
            spam!().spam(spam!());
            for _ in spam!() {}
            || spam!();
            while spam!() {}
            break spam!();
            return spam!();
            match spam!() {
                _ if spam!() => spam!(),
            }
            spam!()(spam!());
            Spam { spam: spam!() };
            spam!()[spam!()];
            await spam!();
            spam!() as usize;
            &spam!();
            -spam!();
            spam!()..spam!();
            spam!() + spam!();
        }
        "#,
        expect![[r#"
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            39..442 '{     ...!(); }': ()
            73..94 'spam!(...am!())': {unknown}
            100..119 'for _ ...!() {}': fn into_iter<isize>(isize) -> <isize as IntoIterator>::IntoIter
            100..119 'for _ ...!() {}': IntoIterator::IntoIter<isize>
            100..119 'for _ ...!() {}': !
            100..119 'for _ ...!() {}': IntoIterator::IntoIter<isize>
            100..119 'for _ ...!() {}': &mut IntoIterator::IntoIter<isize>
            100..119 'for _ ...!() {}': fn next<IntoIterator::IntoIter<isize>>(&mut IntoIterator::IntoIter<isize>) -> Option<<IntoIterator::IntoIter<isize> as Iterator>::Item>
            100..119 'for _ ...!() {}': Option<Iterator::Item<IntoIterator::IntoIter<isize>>>
            100..119 'for _ ...!() {}': ()
            100..119 'for _ ...!() {}': ()
            100..119 'for _ ...!() {}': ()
            104..105 '_': Iterator::Item<IntoIterator::IntoIter<isize>>
            117..119 '{}': ()
            124..134 '|| spam!()': impl Fn() -> isize
            140..156 'while ...!() {}': ()
            154..156 '{}': ()
            161..174 'break spam!()': !
            180..194 'return spam!()': !
            200..254 'match ...     }': isize
            224..225 '_': isize
            259..275 'spam!(...am!())': {unknown}
            281..303 'Spam {...m!() }': {unknown}
            309..325 'spam!(...am!()]': {unknown}
            350..366 'spam!(... usize': usize
            372..380 '&spam!()': &isize
            386..394 '-spam!()': isize
            400..416 'spam!(...pam!()': {unknown}
            422..439 'spam!(...pam!()': isize
        "#]],
    );
}

#[test]
fn expr_macro_rules_expanded_in_various_places() {
    check_infer(
        r#"
        //- minicore: iterator
        macro_rules! spam {
            () => (1isize);
        }

        fn spam() {
            spam!();
            (spam!());
            spam!().spam(spam!());
            for _ in spam!() {}
            || spam!();
            while spam!() {}
            break spam!();
            return spam!();
            match spam!() {
                _ if spam!() => spam!(),
            }
            spam!()(spam!());
            Spam { spam: spam!() };
            spam!()[spam!()];
            await spam!();
            spam!() as usize;
            &spam!();
            -spam!();
            spam!()..spam!();
            spam!() + spam!();
        }
        "#,
        expect![[r#"
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            !0..6 '1isize': isize
            53..456 '{     ...!(); }': ()
            87..108 'spam!(...am!())': {unknown}
            114..133 'for _ ...!() {}': fn into_iter<isize>(isize) -> <isize as IntoIterator>::IntoIter
            114..133 'for _ ...!() {}': IntoIterator::IntoIter<isize>
            114..133 'for _ ...!() {}': !
            114..133 'for _ ...!() {}': IntoIterator::IntoIter<isize>
            114..133 'for _ ...!() {}': &mut IntoIterator::IntoIter<isize>
            114..133 'for _ ...!() {}': fn next<IntoIterator::IntoIter<isize>>(&mut IntoIterator::IntoIter<isize>) -> Option<<IntoIterator::IntoIter<isize> as Iterator>::Item>
            114..133 'for _ ...!() {}': Option<Iterator::Item<IntoIterator::IntoIter<isize>>>
            114..133 'for _ ...!() {}': ()
            114..133 'for _ ...!() {}': ()
            114..133 'for _ ...!() {}': ()
            118..119 '_': Iterator::Item<IntoIterator::IntoIter<isize>>
            131..133 '{}': ()
            138..148 '|| spam!()': impl Fn() -> isize
            154..170 'while ...!() {}': ()
            168..170 '{}': ()
            175..188 'break spam!()': !
            194..208 'return spam!()': !
            214..268 'match ...     }': isize
            238..239 '_': isize
            273..289 'spam!(...am!())': {unknown}
            295..317 'Spam {...m!() }': {unknown}
            323..339 'spam!(...am!()]': {unknown}
            364..380 'spam!(... usize': usize
            386..394 '&spam!()': &isize
            400..408 '-spam!()': isize
            414..430 'spam!(...pam!()': {unknown}
            436..453 'spam!(...pam!()': isize
        "#]],
    );
}

#[test]
fn expr_macro_expanded_in_stmts() {
    check_infer(
        r#"
        macro_rules! id { ($($es:tt)*) => { $($es)* } }
        fn foo() {
            id! { let a = (); }
        }
        "#,
        expect![[r#"
            !3..4 'a': ()
            !5..7 '()': ()
            57..84 '{     ...); } }': ()
        "#]],
    );
}

#[test]
fn recursive_macro_expanded_in_stmts() {
    check_infer(
        r#"
        macro_rules! ng {
            ([$($tts:tt)*]) => {
                $($tts)*;
            };
            ([$($tts:tt)*] $head:tt $($rest:tt)*) => {
                ng! {
                    [$($tts)* $head] $($rest)*
                }
            };
        }
        fn foo() {
            ng!([] let a = 3);
            let b = a;
        }
        "#,
        expect![[r#"
            !3..4 'a': i32
            !5..6 '3': i32
            196..237 '{     ...= a; }': ()
            229..230 'b': i32
            233..234 'a': i32
        "#]],
    );
}

#[test]
fn recursive_inner_item_macro_rules() {
    check_infer(
        r#"
        macro_rules! mac {
            () => { mac!($)};
            ($x:tt) => { macro_rules! blub { () => { 1 }; } };
        }
        fn foo() {
            mac!();
            let a = blub!();
        }
        "#,
        expect![[r#"
            !0..1 '1': i32
            107..143 '{     ...!(); }': ()
            129..130 'a': i32
        "#]],
    );
}

#[test]
fn infer_macro_defining_block_with_items() {
    check_infer(
        r#"
        macro_rules! foo {
            () => {{
                fn bar() -> usize { 0 }
                bar()
            }};
        }
        fn main() {
            let _a = foo!();
        }
    "#,
        expect![[r#"
            !15..18 '{0}': usize
            !16..17 '0': usize
            !0..24 '{fnbar...bar()}': usize
            !18..21 'bar': fn bar() -> usize
            !18..23 'bar()': usize
            98..122 '{     ...!(); }': ()
            108..110 '_a': usize
        "#]],
    );
}

#[test]
fn infer_type_value_macro_having_same_name() {
    check_infer(
        r#"
        #[macro_export]
        macro_rules! foo {
            () => {
                mod foo {
                    pub use super::foo;
                }
            };
            ($x:tt) => {
                $x
            };
        }

        foo!();

        fn foo() {
            let foo = foo::foo!(42i32);
        }
        "#,
        expect![[r#"
            !0..5 '42i32': i32
            170..205 '{     ...32); }': ()
            180..183 'foo': i32
        "#]],
    );
}

#[test]
fn processes_impls_generated_by_macros() {
    check_types(
        r#"
macro_rules! m {
    ($ident:ident) => (impl Trait for $ident {})
}
trait Trait { fn foo(self) -> u128 { 0 } }
struct S;
m!(S);
fn test() { S.foo(); }
          //^^^^^^^ u128
"#,
    );
}

#[test]
fn infer_assoc_items_generated_by_macros() {
    check_types(
        r#"
macro_rules! m {
    () => (fn foo(&self) -> u128 {0})
}
struct S;
impl S {
    m!();
}

fn test() { S.foo(); }
          //^^^^^^^ u128
"#,
    );
}

#[test]
fn infer_assoc_items_generated_by_macros_chain() {
    check_types(
        r#"
macro_rules! m_inner {
    () => {fn foo(&self) -> u128 {0}}
}
macro_rules! m {
    () => {m_inner!();}
}

struct S;
impl S {
    m!();
}

fn test() { S.foo(); }
          //^^^^^^^ u128
"#,
    );
}

#[test]
fn infer_macro_with_dollar_crate_is_correct_in_expr() {
    check_types(
        r#"
//- /main.rs crate:main deps:foo
fn test() {
    let x = (foo::foo!(1), foo::foo!(2));
    x;
} //^ (i32, usize)

//- /lib.rs crate:foo
#[macro_export]
macro_rules! foo {
    (1) => { $crate::bar!() };
    (2) => { 1 + $crate::baz() };
}

#[macro_export]
macro_rules! bar {
    () => { 42 }
}

pub fn baz() -> usize { 31usize }
"#,
    );
}

#[test]
fn infer_macro_with_dollar_crate_is_correct_in_trait_associate_type() {
    check_types(
        r#"
//- /main.rs crate:main deps:foo
use foo::Trait;

fn test() {
    let msg = foo::Message(foo::MessageRef);
    let r = msg.deref();
    r;
  //^ &MessageRef
}

//- /lib.rs crate:foo
pub struct MessageRef;
pub struct Message(MessageRef);

pub trait Trait {
    type Target;
    fn deref(&self) -> &Self::Target;
}

#[macro_export]
macro_rules! expand {
    () => {
        impl Trait for Message {
            type Target = $crate::MessageRef;
            fn deref(&self) ->  &Self::Target {
                &self.0
            }
        }
    }
}

expand!();
"#,
    );
}

#[test]
fn infer_macro_with_dollar_crate_in_def_site() {
    check_types(
        r#"
//- /main.rs crate:main deps:foo
use foo::expand;

macro_rules! list {
    ($($tt:tt)*) => { $($tt)* }
}

fn test() {
    let r = expand!();
    r;
  //^ u128
}

//- /lib.rs crate:foo
#[macro_export]
macro_rules! expand {
    () => { list!($crate::m!()) };
}

#[macro_export]
macro_rules! m {
    () => { 0u128 };
}
"#,
    );
}

#[test]
fn infer_type_value_non_legacy_macro_use_as() {
    check_infer(
        r#"
        mod m {
            macro_rules! _foo {
                ($x:ident) => { type $x = u64; }
            }
            pub(crate) use _foo as foo;
        }

        m::foo!(foo);
        use foo as bar;
        fn f() -> bar { 0 }
        fn main() {
            let _a  = f();
        }
        "#,
        expect![[r#"
            158..163 '{ 0 }': u64
            160..161 '0': u64
            174..196 '{     ...f(); }': ()
            184..186 '_a': u64
            190..191 'f': fn f() -> u64
            190..193 'f()': u64
        "#]],
    );
}

#[test]
fn infer_local_macro() {
    check_infer(
        r#"
        fn main() {
            macro_rules! foo {
                () => { 1usize }
            }
            let _a  = foo!();
        }
        "#,
        expect![[r#"
            !0..6 '1usize': usize
            10..89 '{     ...!(); }': ()
            74..76 '_a': usize
        "#]],
    );
}

#[test]
fn infer_local_inner_macros() {
    check_types(
        r#"
//- /main.rs crate:main deps:foo
fn test() {
    let x = foo::foo!(1);
    x;
} //^ i32

//- /lib.rs crate:foo
#[macro_export(local_inner_macros)]
macro_rules! foo {
    (1) => { bar!() };
}

#[macro_export]
macro_rules! bar {
    () => { 42 }
}

"#,
    );
}

#[test]
fn infer_builtin_macros_line() {
    check_infer(
        r#"
        #[rustc_builtin_macro]
        macro_rules! line {() => {}}

        fn main() {
            let x = line!();
        }
        "#,
        expect![[r#"
            !0..1 '0': i32
            !0..6 '0asu32': u32
            63..87 '{     ...!(); }': ()
            73..74 'x': u32
        "#]],
    );
}

#[test]
fn infer_builtin_macros_file() {
    check_infer(
        r#"
        #[rustc_builtin_macro]
        macro_rules! file {() => {}}

        fn main() {
            let x = file!();
        }
        "#,
        expect![[r#"
            !0..2 '""': &str
            63..87 '{     ...!(); }': ()
            73..74 'x': &str
        "#]],
    );
}

#[test]
fn infer_builtin_macros_column() {
    check_infer(
        r#"
        #[rustc_builtin_macro]
        macro_rules! column {() => {}}

        fn main() {
            let x = column!();
        }
        "#,
        expect![[r#"
            !0..1 '0': i32
            !0..6 '0asu32': u32
            65..91 '{     ...!(); }': ()
            75..76 'x': u32
        "#]],
    );
}

#[test]
fn infer_builtin_macros_concat() {
    check_infer(
        r#"
        #[rustc_builtin_macro]
        macro_rules! concat {() => {}}

        fn main() {
            let x = concat!("hello", concat!("world", "!"));
        }
        "#,
        expect![[r#"
            !0..13 '"helloworld!"': &str
            65..121 '{     ...")); }': ()
            75..76 'x': &str
        "#]],
    );
}

#[test]
fn infer_builtin_macros_include() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    bar();
} //^^^^^ u32

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

#[test]
fn infer_builtin_macros_include_expression() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}
fn main() {
    let i = include!("bla.rs");
    i;
  //^ i32
}
//- /bla.rs
0
        "#,
    )
}

#[test]
fn infer_builtin_macros_include_child_mod() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("f/foo.rs");

fn main() {
    bar::bar();
} //^^^^^^^^^^ u32

//- /f/foo.rs
pub mod bar;

//- /f/bar.rs
pub fn bar() -> u32 {0}
"#,
    );
}

#[test]
fn infer_builtin_macros_include_str() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include_str {() => {}}

fn main() {
    let a = include_str!("foo.rs");
    a;
} //^ &str

//- /foo.rs
hello
"#,
    );
}

#[test]
fn infer_builtin_macros_include_str_with_lazy_nested() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! concat {() => {}}
#[rustc_builtin_macro]
macro_rules! include_str {() => {}}

macro_rules! m {
    ($x:expr) => {
        concat!("foo", $x)
    };
}

fn main() {
    let a = include_str!(m!(".rs"));
    a;
} //^ &str

//- /foo.rs
hello
"#,
    );
}

#[test]
fn benchmark_include_macro() {
    if skip_slow_tests() {
        return;
    }
    let data = bench_fixture::big_struct();
    let fixture = r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    RegisterBlock { };
  //^^^^^^^^^^^^^^^^^ RegisterBlock
}
    "#;
    let fixture = format!("{fixture}\n//- /foo.rs\n{data}");

    {
        let _b = bench("include macro");
        check_types(&fixture);
    }
}

#[test]
fn infer_builtin_macros_include_concat() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

include!(concat!("f", "oo.rs"));

fn main() {
    bar();
} //^^^^^ u32

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

#[test]
fn infer_builtin_macros_include_concat_with_bad_env_should_failed() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

#[rustc_builtin_macro]
macro_rules! env {() => {}}

include!(concat!(env!("OUT_DIR"), "/foo.rs"));

fn main() {
    bar();
} //^^^^^ {unknown}

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

#[test]
fn infer_builtin_macros_include_itself_should_failed() {
    check_types(
        r#"
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("main.rs");

fn main() {
    0;
} //^ i32
"#,
    );
}

#[test]
fn infer_builtin_macros_concat_with_lazy() {
    check_infer(
        r#"
        macro_rules! hello {() => {"hello"}}

        #[rustc_builtin_macro]
        macro_rules! concat {() => {}}

        fn main() {
            let x = concat!(hello!(), concat!("world", "!"));
        }
        "#,
        expect![[r#"
            !0..13 '"helloworld!"': &str
            103..160 '{     ...")); }': ()
            113..114 'x': &str
        "#]],
    );
}

#[test]
fn infer_builtin_macros_env() {
    check_types(
        r#"
        //- /main.rs env:foo=bar
        #[rustc_builtin_macro]
        macro_rules! env {() => {}}

        fn main() {
            let x = env!("foo");
              //^ &str
        }
        "#,
    );
}

#[test]
fn infer_builtin_macros_option_env() {
    check_types(
        r#"
        //- minicore: option
        //- /main.rs env:foo=bar
        #[rustc_builtin_macro]
        macro_rules! option_env {() => {}}

        fn main() {
            let x = option_env!("foo");
              //^ Option<&str>
        }
        "#,
    );
}

#[test]
fn infer_derive_clone_simple() {
    check_types(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct S;
fn test() {
    S.clone();
} //^^^^^^^^^ S
"#,
    );
}

#[test]
fn infer_derive_clone_with_params() {
    check_types(
        r#"
//- minicore: clone, derive
#[derive(Clone)]
struct S;
#[derive(Clone)]
struct Wrapper<T>(T);
struct NonClone;
fn test() {
    let x = (Wrapper(S).clone(), Wrapper(NonClone).clone());
    x;
  //^ (Wrapper<S>, {unknown})
}
"#,
    );
}

#[test]
fn infer_custom_derive_simple() {
    // FIXME: this test current now do nothing
    check_types(
        r#"
//- minicore: derive
use foo::Foo;

#[derive(Foo)]
struct S{}

fn test() {
    S{};
} //^^^ S
"#,
    );
}

#[test]
fn macro_in_arm() {
    check_infer(
        r#"
        macro_rules! unit {
            () => { () };
        }

        fn main() {
            let x = match () {
                unit!() => 92u32,
            };
        }
        "#,
        expect![[r#"
            !0..2 '()': ()
            51..110 '{     ...  }; }': ()
            61..62 'x': u32
            65..107 'match ...     }': u32
            71..73 '()': ()
            95..100 '92u32': u32
        "#]],
    );
}

#[test]
fn macro_in_type_alias_position() {
    check_infer(
        r#"
        macro_rules! U32 {
            () => { u32 };
        }

        trait Foo {
            type Ty;
        }

        impl<T> Foo for T {
            type Ty = U32!();
        }

        type TayTo = U32!();

        fn testy() {
            let a: <() as Foo>::Ty;
            let b: TayTo;
        }
        "#,
        expect![[r#"
            147..196 '{     ...yTo; }': ()
            157..158 'a': u32
            185..186 'b': u32
        "#]],
    );
}

#[test]
fn nested_macro_in_type_alias_position() {
    check_infer(
        r#"
        macro_rules! U32Inner2 {
            () => { u32 };
        }

        macro_rules! U32Inner1 {
            () => { U32Inner2!() };
        }

        macro_rules! U32 {
            () => { U32Inner1!() };
        }

        trait Foo {
            type Ty;
        }

        impl<T> Foo for T {
            type Ty = U32!();
        }

        type TayTo = U32!();

        fn testy() {
            let a: <() as Foo>::Ty;
            let b: TayTo;
        }
        "#,
        expect![[r#"
            259..308 '{     ...yTo; }': ()
            269..270 'a': u32
            297..298 'b': u32
        "#]],
    );
}

#[test]
fn macros_in_type_alias_position_generics() {
    check_infer(
        r#"
        struct Foo<A, B>(A, B);

        macro_rules! U32 {
            () => { u32 };
        }

        macro_rules! Bar {
            () => { Foo<U32!(), U32!()> };
        }

        trait Moo {
            type Ty;
        }

        impl<T> Moo for T {
            type Ty = Bar!();
        }

        type TayTo = Bar!();

        fn main() {
            let a: <() as Moo>::Ty;
            let b: TayTo;
        }
        "#,
        expect![[r#"
            228..277 '{     ...yTo; }': ()
            238..239 'a': Foo<u32, u32>
            266..267 'b': Foo<u32, u32>
        "#]],
    );
}

#[test]
fn macros_in_type_position() {
    check_infer(
        r#"
        struct Foo<A, B>(A, B);

        macro_rules! U32 {
            () => { u32 };
        }

        macro_rules! Bar {
            () => { Foo<U32!(), U32!()> };
        }

        fn main() {
            let a: Bar!();
        }
        "#,
        expect![[r#"
            133..155 '{     ...!(); }': ()
            143..144 'a': Foo<u32, u32>
        "#]],
    );
}

#[test]
fn macros_in_type_generics() {
    check_infer(
        r#"
        struct Foo<A, B>(A, B);

        macro_rules! U32 {
            () => { u32 };
        }

        macro_rules! Bar {
            () => { Foo<U32!(), U32!()> };
        }

        trait Moo {
            type Ty;
        }

        impl<T> Moo for T {
            type Ty = Foo<Bar!(), Bar!()>;
        }

        type TayTo = Foo<Bar!(), U32!()>;

        fn main() {
            let a: <() as Moo>::Ty;
            let b: TayTo;
        }
        "#,
        expect![[r#"
            254..303 '{     ...yTo; }': ()
            264..265 'a': Foo<Foo<u32, u32>, Foo<u32, u32>>
            292..293 'b': Foo<Foo<u32, u32>, u32>
        "#]],
    );
}

#[test]
fn infinitely_recursive_macro_type() {
    check_infer(
        r#"
        struct Bar<T, X>(T, X);

        macro_rules! Foo {
            () => { Foo!() }
        }

        macro_rules! U32 {
            () => { u32 }
        }

        type A = Foo!();
        type B = Bar<Foo!(), U32!()>;

        fn main() {
            let a: A;
            let b: B;
        }
        "#,
        expect![[r#"
            166..197 '{     ...: B; }': ()
            176..177 'a': {unknown}
            190..191 'b': Bar<{unknown}, u32>
        "#]],
    );
}

#[test]
fn cfg_tails() {
    check_infer_with_mismatches(
        r#"
//- /lib.rs crate:foo cfg:feature=foo
struct S {}

impl S {
    fn new2(bar: u32) -> Self {
        #[cfg(feature = "foo")]
        { Self { } }
        #[cfg(not(feature = "foo"))]
        { Self { } }
    }
}
"#,
        expect![[r#"
            34..37 'bar': u32
            52..170 '{     ...     }': S
            62..106 '#[cfg(... { } }': S
            96..104 'Self { }': S
        "#]],
    );
}

#[test]
fn infer_in_unexpandable_attr_proc_macro_1() {
    check_types(
        r#"
//- /main.rs crate:main deps:mac
#[mac::attr_macro]
fn foo() {
    let xxx = 1;
      //^^^ i32
}

//- /mac.rs crate:mac
#![crate_type="proc-macro"]
#[proc_macro_attribute]
pub fn attr_macro() {}
"#,
    );
}

#[test]
fn infer_in_unexpandable_attr_proc_macro_in_impl() {
    check_types(
        r#"
//- /main.rs crate:main deps:mac
struct Foo;
impl Foo {
    #[mac::attr_macro]
    fn foo() {
        let xxx = 1;
          //^^^ i32
    }
}

//- /mac.rs crate:mac
#![crate_type="proc-macro"]
#[proc_macro_attribute]
pub fn attr_macro() {}
"#,
    );
}

#[test]
fn infer_in_unexpandable_attr_proc_macro_in_trait() {
    check_types(
        r#"
//- /main.rs crate:main deps:mac
trait Foo {
    #[mac::attr_macro]
    fn foo() {
        let xxx = 1;
          //^^^ i32
    }
}

//- /mac.rs crate:mac
#![crate_type="proc-macro"]
#[proc_macro_attribute]
pub fn attr_macro() {}
"#,
    );
}
