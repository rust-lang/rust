use expect_test::{Expect, expect};
use hir_def::{
    AdtId, GenericDefId, ModuleDefId, hir::generics::GenericParamDataRef, src::HasSource,
};
use itertools::Itertools;
use rustc_type_ir::Variance;
use stdx::format_to;
use syntax::{AstNode, ast::HasName};
use test_fixture::WithFixture;

use hir_def::Lookup;

use crate::{db::HirDatabase, generics::generics, test_db::TestDB};

#[test]
fn phantom_data() {
    check(
        r#"
//- minicore: phantom_data

struct Covariant<A> {
    t: core::marker::PhantomData<A>
}
"#,
        expect![[r#"
                Covariant[A: covariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_types() {
    check(
        r#"
//- minicore: cell
#![feature(lang_items)]

use core::cell::UnsafeCell;

struct InvariantMut<'a,A:'a,B:'a> { //~ ERROR ['a: +, A: o, B: o]
    t: &'a mut (A,B)
}

struct InvariantCell<A> { //~ ERROR [A: o]
    t: UnsafeCell<A>
}

struct InvariantIndirect<A> { //~ ERROR [A: o]
    t: InvariantCell<A>
}

struct Covariant<A> { //~ ERROR [A: +]
    t: A, u: fn() -> A
}

struct Contravariant<A> { //~ ERROR [A: -]
    t: fn(A)
}

enum Enum<A,B,C> { //~ ERROR [A: +, B: -, C: o]
    Foo(Covariant<A>),
    Bar(Contravariant<B>),`
    Zed(Covariant<C>,Contravariant<C>)
}

#[repr(transparent)]
#[lang = "covariant_unsafe_cell"]
pub struct CovariantUnsafeCell<T: ?Sized>(UnsafeCell<T>); //~ ERROR [T: +]
"#,
        expect![[r#"
                InvariantMut['a: covariant, A: invariant, B: invariant]
                InvariantCell[A: invariant]
                InvariantIndirect[A: invariant]
                Covariant[A: covariant]
                Contravariant[A: contravariant]
                Enum[A: covariant, B: contravariant, C: invariant]
                CovariantUnsafeCell[T: covariant]
            "#]],
    );
}

#[test]
fn type_resolve_error_two_structs_deep() {
    check(
        r#"
struct Hello<'a> {
    missing: Missing<'a>,
}

struct Other<'a> {
    hello: Hello<'a>,
}
"#,
        expect![[r#"
                Hello['a: bivariant]
                Other['a: bivariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_associated_consts() {
    check(
        r#"
trait Trait {
    const Const: usize;
}

struct Foo<T: Trait> {
    field: [u8; <T as Trait>::Const]
}
"#,
        expect![[r#"
            Foo[T: bivariant]
        "#]],
    );
}

#[test]
fn rustc_test_variance_associated_types() {
    check(
        r#"
trait Trait<'a> {
    type Type;

    fn method(&'a self) { }
}

struct Foo<'a, T : Trait<'a>> { //~ ERROR ['a: +, T: +]
    field: (T, &'a ())
}

struct Bar<'a, T : Trait<'a>> { //~ ERROR ['a: o, T: o]
    field: <T as Trait<'a>>::Type
}

"#,
        expect![[r#"
                method[Self: contravariant, 'a: contravariant]
                Foo['a: covariant, T: covariant]
                Bar['a: invariant, T: invariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_associated_types2() {
    // FIXME: RPITs have variance, but we can't treat them as their own thing right now
    check(
        r#"
trait Foo {
    type Bar;
}

fn make() -> *const dyn Foo<Bar = &'static u32> {}
"#,
        expect![""],
    );
}

#[test]
fn rustc_test_variance_trait_bounds() {
    check(
        r#"
trait Getter<T> {
    fn get(&self) -> T;
}

trait Setter<T> {
    fn get(&self, _: T);
}

struct TestStruct<U,T:Setter<U>> { //~ ERROR [U: +, T: +]
    t: T, u: U
}

enum TestEnum<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    Foo(T)
}

struct TestContraStruct<U,T:Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}

struct TestBox<U,T:Getter<U>+Setter<U>> { //~ ERROR [U: *, T: +]
    //~^ ERROR: `U` is never used
    t: T
}
"#,
        expect![[r#"
                get[Self: contravariant, T: covariant]
                get[Self: contravariant, T: contravariant]
                TestStruct[U: covariant, T: covariant]
                TestEnum[U: bivariant, T: covariant]
                TestContraStruct[U: bivariant, T: covariant]
                TestBox[U: bivariant, T: covariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_trait_matching() {
    check(
        r#"

trait Get<T> {
    fn get(&self) -> T;
}

struct Cloner<T:Clone> {
    t: T
}

impl<T:Clone> Get<T> for Cloner<T> {
    fn get(&self) -> T {}
}

fn get<'a, G>(get: &G) -> i32
    where G : Get<&'a i32>
{}

fn pick<'b, G>(get: &'b G, if_odd: &'b i32) -> i32
    where G : Get<&'b i32>
{}
"#,
        expect![[r#"
                get[Self: contravariant, T: covariant]
                Cloner[T: covariant]
                get[T: invariant]
                get['a: invariant, G: contravariant]
                pick['b: contravariant, G: contravariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_trait_object_bound() {
    check(
        r#"
enum Option<T> {
    Some(T),
    None
}
trait T { fn foo(&self); }

struct TOption<'a> { //~ ERROR ['a: +]
    v: Option<*const (dyn T + 'a)>,
}
"#,
        expect![[r#"
                Option[T: covariant]
                foo[Self: contravariant]
                TOption['a: covariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_types_bounds() {
    check(
        r#"
//- minicore: send
struct TestImm<A, B> { //~ ERROR [A: +, B: +]
    x: A,
    y: B,
}

struct TestMut<A, B:'static> { //~ ERROR [A: +, B: o]
    x: A,
    y: &'static mut B,
}

struct TestIndirect<A:'static, B:'static> { //~ ERROR [A: +, B: o]
    m: TestMut<A, B>
}

struct TestIndirect2<A:'static, B:'static> { //~ ERROR [A: o, B: o]
    n: TestMut<A, B>,
    m: TestMut<B, A>
}

trait Getter<A> {
    fn get(&self) -> A;
}

trait Setter<A> {
    fn set(&mut self, a: A);
}

struct TestObject<A, R> { //~ ERROR [A: o, R: o]
    n: *const (dyn Setter<A> + Send),
    m: *const (dyn Getter<R> + Send),
}
"#,
        expect![[r#"
                TestImm[A: covariant, B: covariant]
                TestMut[A: covariant, B: invariant]
                TestIndirect[A: covariant, B: invariant]
                TestIndirect2[A: invariant, B: invariant]
                get[Self: contravariant, A: covariant]
                set[Self: invariant, A: contravariant]
                TestObject[A: invariant, R: invariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_unused_region_param() {
    check(
        r#"
struct SomeStruct<'a> { x: u32 } //~ ERROR parameter `'a` is never used
enum SomeEnum<'a> { Nothing } //~ ERROR parameter `'a` is never used
trait SomeTrait<'a> { fn foo(&self); } // OK on traits.
"#,
        expect![[r#"
                SomeStruct['a: bivariant]
                SomeEnum['a: bivariant]
                foo[Self: contravariant, 'a: invariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_unused_type_param() {
    check(
        r#"
//- minicore: sized
struct SomeStruct<A> { x: u32 }
enum SomeEnum<A> { Nothing }
enum ListCell<T> {
    Cons(*const ListCell<T>),
    Nil
}

struct SelfTyAlias<T>(*const Self);
struct WithBounds<T: Sized> {}
struct WithWhereBounds<T> where T: Sized {}
struct WithOutlivesBounds<T: 'static> {}
struct DoubleNothing<T> {
    s: SomeStruct<T>,
}

"#,
        expect![[r#"
                SomeStruct[A: bivariant]
                SomeEnum[A: bivariant]
                ListCell[T: bivariant]
                SelfTyAlias[T: bivariant]
                WithBounds[T: bivariant]
                WithWhereBounds[T: bivariant]
                WithOutlivesBounds[T: bivariant]
                DoubleNothing[T: bivariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_use_contravariant_struct1() {
    check(
        r#"
struct SomeStruct<T>(fn(T));

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{}
"#,
        expect![[r#"
                SomeStruct[T: contravariant]
                foo['min: contravariant, 'max: covariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_use_contravariant_struct2() {
    check(
        r#"
struct SomeStruct<T>(fn(T));

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{}
"#,
        expect![[r#"
                SomeStruct[T: contravariant]
                bar['min: covariant, 'max: contravariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_use_covariant_struct1() {
    check(
        r#"
struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{}
"#,
        expect![[r#"
                SomeStruct[T: covariant]
                foo['min: contravariant, 'max: covariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_use_covariant_struct2() {
    check(
        r#"
struct SomeStruct<T>(T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{}
"#,
        expect![[r#"
                SomeStruct[T: covariant]
                foo['min: covariant, 'max: contravariant]
            "#]],
    );
}

#[test]
fn rustc_test_variance_use_invariant_struct1() {
    check(
        r#"
struct SomeStruct<T>(*mut T);

fn foo<'min,'max>(v: SomeStruct<&'max ()>)
                  -> SomeStruct<&'min ()>
    where 'max : 'min
{}

fn bar<'min,'max>(v: SomeStruct<&'min ()>)
                  -> SomeStruct<&'max ()>
    where 'max : 'min
{}
"#,
        expect![[r#"
                SomeStruct[T: invariant]
                foo['min: invariant, 'max: invariant]
                bar['min: invariant, 'max: invariant]
            "#]],
    );
}

#[test]
fn invalid_arg_counts() {
    check(
        r#"
struct S<T>(T);
struct S2<T>(S<>);
struct S3<T>(S<T, T>);
"#,
        expect![[r#"
                S[T: covariant]
                S2[T: bivariant]
                S3[T: covariant]
            "#]],
    );
}

#[test]
fn prove_fixedpoint() {
    check(
        r#"
struct FixedPoint<T, U, V>(&'static FixedPoint<(), T, U>, V);
"#,
        expect![[r#"
                FixedPoint[T: covariant, U: covariant, V: covariant]
            "#]],
    );
}

#[track_caller]
fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected: Expect) {
    // use tracing_subscriber::{layer::SubscriberExt, Layer};
    // let my_layer = tracing_subscriber::fmt::layer();
    // let _g = tracing::subscriber::set_default(tracing_subscriber::registry().with(
    //     my_layer.with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
    //         metadata.target().starts_with("hir_ty::variance")
    //     })),
    // ));
    let (db, file_id) = TestDB::with_single_file(ra_fixture);

    crate::attach_db(&db, || {
        let mut defs: Vec<GenericDefId> = Vec::new();
        let module = db.module_for_file_opt(file_id.file_id(&db)).unwrap();
        let def_map = module.def_map(&db);
        crate::tests::visit_module(&db, def_map, module, &mut |it| {
            defs.push(match it {
                ModuleDefId::FunctionId(it) => it.into(),
                ModuleDefId::AdtId(it) => it.into(),
                ModuleDefId::ConstId(it) => it.into(),
                ModuleDefId::TraitId(it) => it.into(),
                ModuleDefId::TypeAliasId(it) => it.into(),
                _ => return,
            })
        });
        let defs = defs
            .into_iter()
            .filter_map(|def| {
                Some((
                    def,
                    match def {
                        GenericDefId::FunctionId(it) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::AdtId(AdtId::EnumId(it)) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::AdtId(AdtId::StructId(it)) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::AdtId(AdtId::UnionId(it)) => {
                            let loc = it.lookup(&db);
                            loc.source(&db).value.name().unwrap()
                        }
                        GenericDefId::TraitId(_)
                        | GenericDefId::TypeAliasId(_)
                        | GenericDefId::ImplId(_)
                        | GenericDefId::ConstId(_)
                        | GenericDefId::StaticId(_) => return None,
                    },
                ))
            })
            .sorted_by_key(|(_, n)| n.syntax().text_range().start());
        let mut res = String::new();
        for (def, name) in defs {
            let variances = db.variances_of(def);
            if variances.is_empty() {
                continue;
            }
            format_to!(
                res,
                "{name}[{}]\n",
                generics(&db, def)
                    .iter(false)
                    .map(|(_, param)| match param {
                        GenericParamDataRef::TypeParamData(type_param_data) => {
                            type_param_data.name.as_ref().unwrap()
                        }
                        GenericParamDataRef::ConstParamData(const_param_data) =>
                            &const_param_data.name,
                        GenericParamDataRef::LifetimeParamData(lifetime_param_data) => {
                            &lifetime_param_data.name
                        }
                    })
                    .zip_eq(variances)
                    .format_with(", ", |(name, var), f| f(&format_args!(
                        "{}: {}",
                        name.as_str(),
                        match var {
                            Variance::Covariant => "covariant",
                            Variance::Invariant => "invariant",
                            Variance::Contravariant => "contravariant",
                            Variance::Bivariant => "bivariant",
                        },
                    )))
            );
        }

        expected.assert_eq(&res);
    })
}
