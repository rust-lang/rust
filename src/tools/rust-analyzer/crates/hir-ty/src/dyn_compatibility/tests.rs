use std::ops::ControlFlow;

use hir_def::db::DefDatabase;
use rustc_hash::{FxHashMap, FxHashSet};
use syntax::ToSmolStr;
use test_fixture::WithFixture;

use crate::{dyn_compatibility::dyn_compatibility_with_callback, test_db::TestDB};

use super::{
    DynCompatibilityViolation,
    MethodViolationCode::{self, *},
};

use DynCompatibilityViolationKind::*;

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum DynCompatibilityViolationKind {
    SizedSelf,
    SelfReferential,
    Method(MethodViolationCode),
    AssocConst,
    GAT,
    HasNonCompatibleSuperTrait,
}

fn check_dyn_compatibility<'a>(
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expected: impl IntoIterator<Item = (&'a str, Vec<DynCompatibilityViolationKind>)>,
) {
    let mut expected: FxHashMap<_, _> =
        expected.into_iter().map(|(id, osvs)| (id, FxHashSet::from_iter(osvs))).collect();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    for (trait_id, name) in file_ids.into_iter().flat_map(|file_id| {
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id.local_id].scope;
        scope
            .declarations()
            .filter_map(|def| {
                if let hir_def::ModuleDefId::TraitId(trait_id) = def {
                    let name = db
                        .trait_signature(trait_id)
                        .name
                        .display_no_db(file_id.edition(&db))
                        .to_smolstr();
                    Some((trait_id, name))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }) {
        let Some(expected) = expected.remove(name.as_str()) else {
            continue;
        };
        let mut osvs = FxHashSet::default();
        _ = dyn_compatibility_with_callback(&db, trait_id, &mut |osv| {
            osvs.insert(match osv {
                DynCompatibilityViolation::SizedSelf => SizedSelf,
                DynCompatibilityViolation::SelfReferential => SelfReferential,
                DynCompatibilityViolation::Method(_, mvc) => Method(mvc),
                DynCompatibilityViolation::AssocConst(_) => AssocConst,
                DynCompatibilityViolation::GAT(_) => GAT,
                DynCompatibilityViolation::HasNonCompatibleSuperTrait(_) => {
                    HasNonCompatibleSuperTrait
                }
            });
            ControlFlow::Continue(())
        });
        assert_eq!(osvs, expected, "dyn-compatibility violations for `{name}` do not match;");
    }

    let remains: Vec<_> = expected.keys().collect();
    assert!(remains.is_empty(), "Following traits do not exist in the test fixture; {remains:?}");
}

#[test]
fn item_bounds_can_reference_self() {
    check_dyn_compatibility(
        r#"
//- minicore: eq
pub trait Foo {
    type X: PartialEq;
    type Y: PartialEq<Self::Y>;
    type Z: PartialEq<Self::Y>;
}
"#,
        [("Foo", vec![])],
    );
}

#[test]
fn associated_consts() {
    check_dyn_compatibility(
        r#"
trait Bar {
    const X: usize;
}
"#,
        [("Bar", vec![AssocConst])],
    );
}

#[test]
fn bounds_reference_self() {
    check_dyn_compatibility(
        r#"
//- minicore: eq
trait X {
    type U: PartialEq<Self>;
}
"#,
        [("X", vec![SelfReferential])],
    );
}

#[test]
fn by_value_self() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Bar {
    fn bar(self);
}

trait Baz {
    fn baz(self: Self);
}

trait Quux {
    // Legal because of the where clause:
    fn baz(self: Self) where Self : Sized;
}
"#,
        [("Bar", vec![]), ("Baz", vec![]), ("Quux", vec![])],
    );
}

#[test]
fn generic_methods() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Bar {
    fn bar<T>(&self, t: T);
}

trait Quux {
    fn bar<T>(&self, t: T)
        where Self : Sized;
}

trait Qax {
    fn bar<'a>(&self, t: &'a ());
}
"#,
        [("Bar", vec![Method(Generic)]), ("Quux", vec![]), ("Qax", vec![])],
    );
}

#[test]
fn mentions_self() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Bar {
    fn bar(&self, x: &Self);
}

trait Baz {
    fn baz(&self) -> Self;
}

trait Quux {
    fn quux(&self, s: &Self) -> Self where Self : Sized;
}
"#,
        [
            ("Bar", vec![Method(ReferencesSelfInput)]),
            ("Baz", vec![Method(ReferencesSelfOutput)]),
            ("Quux", vec![]),
        ],
    );
}

#[test]
fn no_static() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Foo {
    fn foo() {}
}
"#,
        [("Foo", vec![Method(StaticMethod)])],
    );
}

#[test]
fn sized_self() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Bar: Sized {
    fn bar<T>(&self, t: T);
}
"#,
        [("Bar", vec![SizedSelf])],
    );

    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Bar
    where Self : Sized
{
    fn bar<T>(&self, t: T);
}
"#,
        [("Bar", vec![SizedSelf])],
    );
}

#[test]
fn supertrait_gat() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait GatTrait {
    type Gat<T>;
}

trait SuperTrait<T>: GatTrait {}
"#,
        [("GatTrait", vec![GAT]), ("SuperTrait", vec![HasNonCompatibleSuperTrait])],
    );
}

#[test]
fn supertrait_mentions_self() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Bar<T> {
    fn bar(&self, x: &T);
}

trait Baz : Bar<Self> {
}
"#,
        [("Bar", vec![]), ("Baz", vec![SizedSelf, SelfReferential])],
    );
}

#[test]
fn rustc_issue_19538() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Foo {
    fn foo<T>(&self, val: T);
}

trait Bar: Foo {}
"#,
        [("Foo", vec![Method(Generic)]), ("Bar", vec![HasNonCompatibleSuperTrait])],
    );
}

#[test]
fn rustc_issue_22040() {
    check_dyn_compatibility(
        r#"
//- minicore: fmt, eq, dispatch_from_dyn
use core::fmt::Debug;

trait Expr: Debug + PartialEq {
    fn print_element_count(&self);
}
"#,
        [("Expr", vec![SelfReferential])],
    );
}

#[test]
fn rustc_issue_102762() {
    check_dyn_compatibility(
        r#"
//- minicore: future, send, sync, dispatch_from_dyn, deref
use core::pin::Pin;

struct Box<T: ?Sized> {}
impl<T: ?Sized> core::ops::Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        loop {}
    }
}
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Box<U>> for Box<T> {}

struct Vec<T> {}

pub trait Fetcher: Send + Sync {
    fn get<'a>(self: &'a Box<Self>) -> Pin<Box<dyn Future<Output = Vec<u8>> + 'a>>
    where
        Self: Sync,
    {
        loop {}
    }
}
"#,
        [("Fetcher", vec![Method(UndispatchableReceiver)])],
    );
}

#[test]
fn rustc_issue_102933() {
    check_dyn_compatibility(
        r#"
//- minicore: future, dispatch_from_dyn, deref
use core::future::Future;

struct Box<T: ?Sized> {}
impl<T: ?Sized> core::ops::Deref for Box<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        loop {}
    }
}
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Box<U>> for Box<T> {}

pub trait Service {
    type Response;
    type Future: Future<Output = Self::Response>;
}

pub trait A1: Service<Response = i32> {}

pub trait A2: Service<Future = Box<dyn Future<Output = i32>>> + A1 {
    fn foo(&self) {}
}

pub trait B1: Service<Future = Box<dyn Future<Output = i32>>> {}

pub trait B2: Service<Response = i32> + B1 {
    fn foo(&self) {}
}
        "#,
        [("A2", vec![]), ("B2", vec![])],
    );
}

#[test]
fn rustc_issue_106247() {
    check_dyn_compatibility(
        r#"
//- minicore: sync, dispatch_from_dyn
pub trait Trait {
    fn method(&self) where Self: Sync;
}
"#,
        [("Trait", vec![])],
    );
}

#[test]
fn std_error_is_dyn_compatible() {
    check_dyn_compatibility(
        r#"
//- minicore: fmt, dispatch_from_dyn
trait Erased<'a>: 'a {}

pub struct Request<'a>(dyn Erased<'a> + 'a);

pub trait Error: core::fmt::Debug + core::fmt::Display {
    fn provide<'a>(&'a self, request: &mut Request<'a>);
}
"#,
        [("Error", vec![])],
    );
}

#[test]
fn lifetime_gat_is_dyn_incompatible() {
    check_dyn_compatibility(
        r#"
//- minicore: dispatch_from_dyn
trait Foo {
    type Bar<'a>;
}
"#,
        [("Foo", vec![DynCompatibilityViolationKind::GAT])],
    );
}
