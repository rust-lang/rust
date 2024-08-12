use expect_test::expect;

use super::{check_infer_with_mismatches, check_no_mismatches, check_types};

#[test]
fn associated_type_impl_trait() {
    check_types(
        r#"
trait Foo {}
struct S1;
impl Foo for S1 {}

trait Bar {
    type Item;
    fn bar(&self) -> Self::Item;
}
struct S2;
impl Bar for S2 {
    type Item = impl Foo;
    fn bar(&self) -> Self::Item {
        S1
    }
}

fn test() {
    let x = S2.bar();
      //^ impl Foo + ?Sized
}
        "#,
    );
}

#[test]
fn associated_type_impl_traits_complex() {
    check_types(
        r#"
struct Unary<T>(T);
struct Binary<T, U>(T, U);

trait Foo {}
struct S1;
impl Foo for S1 {}

trait Bar {
    type Item;
    fn bar(&self) -> Unary<Self::Item>;
}
struct S2;
impl Bar for S2 {
    type Item = Unary<impl Foo>;
    fn bar(&self) -> Unary<<Self as Bar>::Item> {
        Unary(Unary(S1))
    }
}

trait Baz {
    type Target1;
    type Target2;
    fn baz(&self) -> Binary<Self::Target1, Self::Target2>;
}
struct S3;
impl Baz for S3 {
    type Target1 = impl Foo;
    type Target2 = Unary<impl Bar>;
    fn baz(&self) -> Binary<Self::Target1, Self::Target2> {
        Binary(S1, Unary(S2))
    }
}

fn test() {
    let x = S3.baz();
      //^ Binary<impl Foo + ?Sized, Unary<impl Bar + ?Sized>>
    let y = x.1.0.bar();
      //^ Unary<Bar::Item<impl Bar + ?Sized>>
}
        "#,
    );
}

#[test]
fn associated_type_with_impl_trait_in_tuple() {
    check_no_mismatches(
        r#"
pub trait Iterator {
    type Item;
}

pub trait Value {}

fn bar<I: Iterator<Item = (usize, impl Value)>>() {}

fn foo() {
    bar();
}
"#,
    );
}

#[test]
fn associated_type_with_impl_trait_in_nested_tuple() {
    check_no_mismatches(
        r#"
pub trait Iterator {
    type Item;
}

pub trait Value {}

fn bar<I: Iterator<Item = ((impl Value, usize), u32)>>() {}

fn foo() {
    bar();
}
"#,
    );
}

#[test]
fn type_alias_impl_trait_simple() {
    check_no_mismatches(
        r#"
trait Trait {}

struct Struct;

impl Trait for Struct {}

type AliasTy = impl Trait;

static ALIAS: AliasTy = {
    let res: AliasTy = Struct;
    res
};
"#,
    );

    check_infer_with_mismatches(
        r#"
trait Trait {}

struct Struct;

impl Trait for Struct {}

type AliasTy = impl Trait;

static ALIAS: i32 = {
    // TATIs cannot be define-used if not in signature or type annotations
    let _a: AliasTy = Struct;
    5
};
"#,
        expect![[r#"
            106..220 '{     ...   5 }': i32
            191..193 '_a': impl Trait + ?Sized
            205..211 'Struct': Struct
            217..218 '5': i32
            205..211: expected impl Trait + ?Sized, got Struct
        "#]],
    )
}
