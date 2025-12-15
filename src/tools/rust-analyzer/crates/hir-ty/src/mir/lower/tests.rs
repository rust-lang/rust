use hir_def::DefWithBodyId;
use test_fixture::WithFixture;

use crate::{db::HirDatabase, setup_tracing, test_db::TestDB};

fn lower_mir(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    crate::attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id].scope;
        let funcs = scope.declarations().filter_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(it) => Some(it),
            _ => None,
        });
        for func in funcs {
            _ = db.mir_body(func.into());
        }
    })
}

#[test]
fn dyn_projection_with_auto_traits_regression_next_solver() {
    lower_mir(
        r#"
//- minicore: sized, send
pub trait Deserializer {}

pub trait Strictest {
    type Object: ?Sized;
}

impl Strictest for dyn CustomValue {
    type Object = dyn CustomValue + Send;
}

pub trait CustomValue: Send {}

impl CustomValue for () {}

struct Box<T: ?Sized>;

type DeserializeFn<T> = fn(&mut dyn Deserializer) -> Box<T>;

fn foo() {
    (|deserializer| Box::new(())) as DeserializeFn<<dyn CustomValue as Strictest>::Object>;
}
    "#,
    );
}

fn check_borrowck(#[rust_analyzer::rust_fixture] ra_fixture: &str) {
    let _tracing = setup_tracing();
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    crate::attach_db(&db, || {
        let file_id = *file_ids.last().unwrap();
        let module_id = db.module_for_file(file_id.file_id(&db));
        let def_map = module_id.def_map(&db);
        let scope = &def_map[module_id].scope;

        let mut bodies: Vec<DefWithBodyId> = Vec::new();

        for decl in scope.declarations() {
            if let hir_def::ModuleDefId::FunctionId(f) = decl {
                bodies.push(f.into());
            }
        }

        for impl_id in scope.impls() {
            let impl_items = impl_id.impl_items(&db);
            for (_, item) in impl_items.items.iter() {
                if let hir_def::AssocItemId::FunctionId(f) = item {
                    bodies.push((*f).into());
                }
            }
        }

        for body in bodies {
            let _ = db.borrowck(body);
        }
    })
}

#[test]
fn regression_21173_const_generic_impl_with_assoc_type() {
    check_borrowck(
        r#"
pub trait Tr {
    type Assoc;
    fn f(&self, handle: Self::Assoc) -> i32;
}

pub struct ConstGeneric<const N: usize>;

impl<const N: usize> Tr for &ConstGeneric<N> {
    type Assoc = AssocTy;

    fn f(&self, a: Self::Assoc) -> i32 {
        a.x
    }
}

pub struct AssocTy {
    x: i32,
}
    "#,
    );
}
