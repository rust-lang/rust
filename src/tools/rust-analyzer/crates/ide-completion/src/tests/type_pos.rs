//! Completion tests for type position.
use expect_test::expect;

use crate::tests::{check, check_with_base_items};

#[test]
fn record_field_ty() {
    check_with_base_items(
        r#"
struct Foo<'lt, T, const C: usize> {
    f: $0
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self   Foo<'_, {unknown}, _>
            st Foo<…> Foo<'_, {unknown}, _>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tp T
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn tuple_struct_field() {
    check_with_base_items(
        r#"
struct Foo<'lt, T, const C: usize>(f$0);
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self   Foo<'_, {unknown}, _>
            st Foo<…> Foo<'_, {unknown}, _>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tp T
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
        "#]],
    )
}

#[test]
fn fn_return_type() {
    check_with_base_items(
        r#"
fn x<'lt, T, const C: usize>() -> $0
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tp T
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn fn_return_type_no_local_items() {
    check_with_base_items(
        r#"
fn foo() -> B$0 {
    struct Bar;
    enum Baz {}
    union Bax {
        i: i32,
        f: f32
    }
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            it ()
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn inferred_type_const() {
    check_with_base_items(
        r#"
struct Foo<T>(T);
const FOO: $0 = Foo(2);
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo<…>        Foo<{unknown}>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            it Foo<i32>
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn inferred_type_closure_param() {
    check_with_base_items(
        r#"
fn f1(f: fn(i32) -> i32) {}
fn f2() {
    f1(|x: $0);
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            it i32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn inferred_type_closure_return() {
    check_with_base_items(
        r#"
fn f1(f: fn(u64) -> u64) {}
fn f2() {
    f1(|x| -> $0 {
        x + 5
    });
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            it u64
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn inferred_type_fn_return() {
    check_with_base_items(
        r#"
fn f2(x: u64) -> $0 {
    x + 5
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            it u64
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn inferred_type_fn_param() {
    check_with_base_items(
        r#"
fn f1(x: i32) {}
fn f2(x: $0) {
    f1(x);
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            it i32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn inferred_type_not_in_the_scope() {
    check_with_base_items(
        r#"
mod a {
    pub struct Foo<T>(T);
    pub fn x() -> Foo<Foo<i32>> {
        Foo(Foo(2))
    }
}
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let foo: $0 = a::x();
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md a
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tp T
            un Union                  Union
            bt u32                      u32
            it a::Foo<a::Foo<i32>>
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn inferred_type_let() {
    check_with_base_items(
        r#"
struct Foo<T>(T);
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let foo: $0 = Foo(2);
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo<…>        Foo<{unknown}>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tp T
            un Union                  Union
            bt u32                      u32
            it Foo<i32>
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn body_type_pos() {
    check_with_base_items(
        r#"
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let _: $0;
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tp T
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
    check_with_base_items(
        r#"
fn foo<'lt, T, const C: usize>() {
    let local = ();
    let _: self::$0;
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
        "#]],
    );
}

#[test]
fn completes_types_and_const_in_arg_list() {
    cov_mark::check!(complete_assoc_type_in_generics_list);
    check_with_base_items(
        r#"
trait Trait1 {
    type Super;
}
trait Trait2: Trait1 {
    type Foo;
}

fn foo<'lt, T: Trait2<$0>, const CONST_PARAM: usize>(_: T) {}
"#,
        expect![[r#"
            ta Foo =  (as Trait2)     type Foo
            ta Super =  (as Trait1) type Super
        "#]],
    );
    check_with_base_items(
        r#"
trait Trait1 {
    type Super;
}
trait Trait2<T>: Trait1 {
    type Foo;
}

fn foo<'lt, T: Trait2<$0>, const CONST_PARAM: usize>(_: T) {}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tt Trait1
            tt Trait2
            tp T
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
    check_with_base_items(
        r#"
trait Trait2<T> {
    type Foo;
}

fn foo<'lt, T: Trait2<self::$0>, const CONST_PARAM: usize>(_: T) {}
    "#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            tt Trait2
            un Union                  Union
        "#]],
    );
}

#[test]
fn no_assoc_completion_outside_type_bounds() {
    check_with_base_items(
        r#"
struct S;
trait Tr<T> {
    type Ty;
}

impl Tr<$0
    "#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            sp Self       dyn Tr<{unknown}>
            st Record                Record
            st S                          S
            st Tuple                  Tuple
            st Unit                    Unit
            tt Tr
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn enum_qualified() {
    check_with_base_items(
        r#"
impl Enum {
    type AssocType = ();
    const ASSOC_CONST: () = ();
    fn assoc_fn() {}
}
fn func(_: Enum::$0) {}
"#,
        expect![[r#"
            ta AssocType type AssocType = ()
        "#]],
    );
}

#[test]
fn completes_type_parameter_or_associated_type() {
    check_with_base_items(
        r#"
trait MyTrait<T, U> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u$0
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt MyTrait
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );

    check_with_base_items(
        r#"
trait MyTrait<T, U> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u$0
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt MyTrait
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );

    check_with_base_items(
        r#"
trait MyTrait<T, U> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u8, I$0
"#,
        expect![[r#"
            ta Item1 =  (as MyTrait) type Item1
            ta Item2 =  (as MyTrait) type Item2
        "#]],
    );
}

#[test]
fn completes_type_parameter_or_associated_type_with_default_value() {
    check_with_base_items(
        r#"
trait MyTrait<T, U = u8> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u$0
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt MyTrait
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );

    check_with_base_items(
        r#"
trait MyTrait<T, U = u8> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u$0
"#,
        expect![[r#"
            en Enum                        Enum
            ma makro!(…)     macro_rules! makro
            md module
            st Record                    Record
            st Tuple                      Tuple
            st Unit                        Unit
            tt MyTrait
            tt Trait
            ta Item1 =  (as MyTrait) type Item1
            ta Item2 =  (as MyTrait) type Item2
            un Union                      Union
            bt u32                          u32
            kw crate::
            kw self::
        "#]],
    );

    check_with_base_items(
        r#"
trait MyTrait<T, U = u8> {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<u8, u8, I$0
"#,
        expect![[r#"
            ta Item1 =  (as MyTrait) type Item1
            ta Item2 =  (as MyTrait) type Item2
        "#]],
    );
}

#[test]
fn completes_types_after_associated_type() {
    check_with_base_items(
        r#"
trait MyTrait {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<Item1 = $0
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt MyTrait
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );

    check_with_base_items(
        r#"
trait MyTrait {
    type Item1;
    type Item2;
};

fn f(t: impl MyTrait<Item1 = u8, Item2 = $0
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt MyTrait
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );

    check_with_base_items(
        r#"
trait MyTrait {
    const C: usize;
};

fn f(t: impl MyTrait<C = $0
"#,
        expect![[r#"
            ct CONST                   Unit
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn type_pos_no_unstable_type_on_stable() {
    check(
        r#"
//- /main.rs crate:main deps:std
use std::*;
struct Foo {
    f: $0
}
//- /std.rs crate:std
#[unstable]
pub struct S;
"#,
        expect![[r#"
            md std
            sp Self Foo
            st Foo  Foo
            bt u32  u32
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn type_pos_unstable_type_on_nightly() {
    check(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
use std::*;
struct Foo {
    f: $0
}
//- /std.rs crate:std
#[unstable]
pub struct S;
"#,
        expect![[r#"
            md std
            sp Self Foo
            st Foo  Foo
            st S      S
            bt u32  u32
            kw crate::
            kw self::
        "#]],
    )
}

#[test]
fn completes_const_and_type_generics_separately() {
    // Function generic params
    check_with_base_items(
        r#"
    struct Foo;
    const X: usize = 0;
    fn foo<T, const N: usize>() {}
    fn main() {
        foo::<F$0, _>();
    }
            "#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo                      Foo
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
    // FIXME: This should probably also suggest completions for types, at least those that have
    // associated constants usable in this position. For example, a user could be typing
    // `foo::<_, { usize::MAX }>()`, but we currently don't suggest `usize` in constant position.
    check_with_base_items(
        r#"
    struct Foo;
    const X: usize = 0;
    fn foo<T, const N: usize>() {}
    fn main() {
        foo::<_, $0>();
    }
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Method generic params
    check_with_base_items(
        r#"
    const X: usize = 0;
    struct Foo;
    impl Foo { fn bar<const N: usize, T>(self) {} }
    fn main() {
        Foo.bar::<_, $0>();
    }
            "#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo                      Foo
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
    check_with_base_items(
        r#"
    const X: usize = 0;
    struct Foo;
    impl Foo { fn bar<const N: usize, T>(self) {} }
    fn main() {
        Foo.bar::<X$0, _>();
    }
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Associated type generic params
    check_with_base_items(
        r#"
    const X: usize = 0;
    struct Foo;
    trait Bar {
        type Baz<T, const X: usize>;
    }
    fn foo(_: impl Bar<Baz<F$0, 0> = ()>) {}
            "#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Foo                      Foo
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Bar
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
    check_with_base_items(
        r#"
    const X: usize = 0;
    struct Foo;
    trait Bar {
        type Baz<T, const X: usize>;
    }
    fn foo<T: Bar<Baz<(), $0> = ()>>() {}
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Type generic params
    check_with_base_items(
        r#"
    const X: usize = 0;
    struct Foo<T, const N: usize>(T);
    fn main() {
        let _: Foo::<_, $0> = Foo(());
    }
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Type alias generic params
    check_with_base_items(
        r#"
    const X: usize = 0;
    struct Foo<T, const N: usize>(T);
    type Bar<const X: usize, U> = Foo<U, X>;
    fn main() {
        let _: Bar::<X$0, _> = Bar(());
    }
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Enum variant params
    check_with_base_items(
        r#"
    const X: usize = 0;
    enum Foo<T, const N: usize> { A(T), B }
    fn main() {
        Foo::B::<(), $0>;
    }
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Trait params
    check_with_base_items(
        r#"
    const X: usize = 0;
    trait Foo<T, const N: usize> {}
    impl Foo<(), $0> for () {}
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Trait alias params
    check_with_base_items(
        r#"
    #![feature(trait_alias)]
    const X: usize = 0;
    trait Foo<T, const N: usize> {}
    trait Bar<const M: usize, U> = Foo<U, M>;
    fn foo<T: Bar<X$0, ()>>() {}
            "#,
        expect![[r#"
            ct CONST                   Unit
            ct X                      usize
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );

    // Omitted lifetime params
    check_with_base_items(
        r#"
struct S<'a, 'b, const C: usize, T>(core::marker::PhantomData<&'a &'b T>);
fn foo<'a>() { S::<F$0, _>; }
        "#,
        expect![[r#"
            ct CONST                   Unit
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );
    // Explicit lifetime params
    check_with_base_items(
        r#"
struct S<'a, 'b, const C: usize, T>(core::marker::PhantomData<&'a &'b T>);
fn foo<'a>() { S::<'static, 'static, F$0, _>; }
        "#,
        expect![[r#"
            ct CONST                   Unit
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );
    check_with_base_items(
        r#"
struct S<'a, 'b, const C: usize, T>(core::marker::PhantomData<&'a &'b T>);
fn foo<'a>() { S::<'static, F$0, _, _>; }
        "#,
        expect![[r#"
            lt 'a
            ma makro!(…) macro_rules! makro
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn complete_traits_on_impl_trait_block() {
    check_with_base_items(
        r#"
trait Foo {}

struct Bar;

impl $0 for Bar { }
"#,
        expect![[r#"
            md module
            tt Foo
            tt Trait
            kw crate::
            kw self::
        "#]],
    );
}

#[test]
fn complete_traits_with_path_on_impl_trait_block() {
    check_with_base_items(
        r#"
mod outer {
    pub trait Foo {}
    pub struct Bar;
    pub mod inner {
    }
}

impl outer::$0 for Bar { }
"#,
        expect![[r#"
            md inner
            tt Foo
        "#]],
    );
}
