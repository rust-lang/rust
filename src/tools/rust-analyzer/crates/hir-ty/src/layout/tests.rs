use std::collections::HashMap;

use base_db::fixture::WithFixture;
use chalk_ir::{AdtId, TyKind};
use hir_def::{
    db::DefDatabase,
    layout::{Layout, LayoutError},
};

use crate::{db::HirDatabase, test_db::TestDB, Interner, Substitution};

use super::layout_of_ty;

fn current_machine_data_layout() -> String {
    project_model::target_data_layout::get(None, None, &HashMap::default()).unwrap()
}

fn eval_goal(ra_fixture: &str, minicore: &str) -> Result<Layout, LayoutError> {
    let target_data_layout = current_machine_data_layout();
    let ra_fixture = format!(
        "{minicore}//- /main.rs crate:test target_data_layout:{target_data_layout}\n{ra_fixture}",
    );

    let (db, file_id) = TestDB::with_single_file(&ra_fixture);
    let module_id = db.module_for_file(file_id);
    let def_map = module_id.def_map(&db);
    let scope = &def_map[module_id.local_id].scope;
    let adt_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::AdtId(x) => {
                let name = match x {
                    hir_def::AdtId::StructId(x) => db.struct_data(x).name.to_smol_str(),
                    hir_def::AdtId::UnionId(x) => db.union_data(x).name.to_smol_str(),
                    hir_def::AdtId::EnumId(x) => db.enum_data(x).name.to_smol_str(),
                };
                (name == "Goal").then_some(x)
            }
            _ => None,
        })
        .unwrap();
    let goal_ty = TyKind::Adt(AdtId(adt_id), Substitution::empty(Interner)).intern(Interner);
    layout_of_ty(&db, &goal_ty, module_id.krate())
}

/// A version of `eval_goal` for types that can not be expressed in ADTs, like closures and `impl Trait`
fn eval_expr(ra_fixture: &str, minicore: &str) -> Result<Layout, LayoutError> {
    let target_data_layout = current_machine_data_layout();
    let ra_fixture = format!(
        "{minicore}//- /main.rs crate:test target_data_layout:{target_data_layout}\nfn main(){{let goal = {{{ra_fixture}}};}}",
    );

    let (db, file_id) = TestDB::with_single_file(&ra_fixture);
    let module_id = db.module_for_file(file_id);
    let def_map = module_id.def_map(&db);
    let scope = &def_map[module_id.local_id].scope;
    let adt_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(x) => {
                let name = db.function_data(x).name.to_smol_str();
                (name == "main").then_some(x)
            }
            _ => None,
        })
        .unwrap();
    let hir_body = db.body(adt_id.into());
    let pat = hir_body
        .pats
        .iter()
        .find(|x| match x.1 {
            hir_def::expr::Pat::Bind { name, .. } => name.to_smol_str() == "goal",
            _ => false,
        })
        .unwrap()
        .0;
    let infer = db.infer(adt_id.into());
    let goal_ty = infer.type_of_pat[pat].clone();
    layout_of_ty(&db, &goal_ty, module_id.krate())
}

#[track_caller]
fn check_size_and_align(ra_fixture: &str, minicore: &str, size: u64, align: u64) {
    let l = eval_goal(ra_fixture, minicore).unwrap();
    assert_eq!(l.size.bytes(), size);
    assert_eq!(l.align.abi.bytes(), align);
}

#[track_caller]
fn check_size_and_align_expr(ra_fixture: &str, minicore: &str, size: u64, align: u64) {
    let l = eval_expr(ra_fixture, minicore).unwrap();
    assert_eq!(l.size.bytes(), size);
    assert_eq!(l.align.abi.bytes(), align);
}

#[track_caller]
fn check_fail(ra_fixture: &str, e: LayoutError) {
    let r = eval_goal(ra_fixture, "");
    assert_eq!(r, Err(e));
}

macro_rules! size_and_align {
    (minicore: $($x:tt),*;$($t:tt)*) => {
        {
            #[allow(dead_code)]
            $($t)*
            check_size_and_align(
                stringify!($($t)*),
                &format!("//- minicore: {}\n", stringify!($($x),*)),
                ::std::mem::size_of::<Goal>() as u64,
                ::std::mem::align_of::<Goal>() as u64,
            );
        }
    };
    ($($t:tt)*) => {
        {
            #[allow(dead_code)]
            $($t)*
            check_size_and_align(
                stringify!($($t)*),
                "",
                ::std::mem::size_of::<Goal>() as u64,
                ::std::mem::align_of::<Goal>() as u64,
            );
        }
    };
}

macro_rules! size_and_align_expr {
    ($($t:tt)*) => {
        {
            #[allow(dead_code)]
            {
                let val = { $($t)* };
                check_size_and_align_expr(
                    stringify!($($t)*),
                    "",
                    ::std::mem::size_of_val(&val) as u64,
                    ::std::mem::align_of_val(&val) as u64,
                );
            }
        }
    };
}

#[test]
fn hello_world() {
    size_and_align! {
        struct Goal(i32);
    }
    size_and_align_expr! {
        2i32
    }
}

#[test]
fn field_order_optimization() {
    size_and_align! {
        struct Goal(u8, i32, u8);
    }
    size_and_align! {
        #[repr(C)]
        struct Goal(u8, i32, u8);
    }
}

#[test]
fn recursive() {
    size_and_align! {
        struct Goal {
            left: &'static Goal,
            right: &'static Goal,
        }
    }
    size_and_align! {
        struct BoxLike<T: ?Sized>(*mut T);
        struct Goal(BoxLike<Goal>);
    }
    check_fail(
        r#"struct Goal(Goal);"#,
        LayoutError::UserError("infinite sized recursive type".to_string()),
    );
    check_fail(
        r#"
        struct Foo<T>(Foo<T>);
        struct Goal(Foo<i32>);
        "#,
        LayoutError::UserError("infinite sized recursive type".to_string()),
    );
}

#[test]
fn generic() {
    size_and_align! {
        struct Pair<A, B>(A, B);
        struct Goal(Pair<Pair<i32, u8>, i64>);
    }
    size_and_align! {
        struct X<const N: usize> {
            field1: [i32; N],
            field2: [u8; N],
        }
        struct Goal(X<1000>);
    }
}

#[test]
fn return_position_impl_trait() {
    size_and_align_expr! {
        trait T {}
        impl T for i32 {}
        impl T for i64 {}
        fn foo() -> impl T { 2i64 }
        foo()
    }
    size_and_align_expr! {
        trait T {}
        impl T for i32 {}
        impl T for i64 {}
        fn foo() -> (impl T, impl T, impl T) { (2i64, 5i32, 7i32) }
        foo()
    }
    size_and_align_expr! {
        struct Foo<T>(T, T, (T, T));
        trait T {}
        impl T for Foo<i32> {}
        impl T for Foo<i64> {}

        fn foo() -> Foo<impl T> { Foo(
            Foo(1i64, 2, (3, 4)),
            Foo(5, 6, (7, 8)),
            (
                Foo(1i64, 2, (3, 4)),
                Foo(5, 6, (7, 8)),
            ),
        ) }
        foo()
    }
}

#[test]
fn enums() {
    size_and_align! {
        enum Goal {
            Quit,
            Move { x: i32, y: i32 },
            ChangeColor(i32, i32, i32),
        }
    }
}

#[test]
fn primitives() {
    size_and_align! {
        struct Goal(i32, i128, isize, usize, f32, f64, bool, char);
    }
}

#[test]
fn tuple() {
    size_and_align! {
        struct Goal((), (i32, u64, bool));
    }
}

#[test]
fn non_zero() {
    size_and_align! {
        minicore: non_zero, option;
        use core::num::NonZeroU8;
        struct Goal(Option<NonZeroU8>);
    }
}

#[test]
fn niche_optimization() {
    size_and_align! {
        minicore: option;
        struct Goal(Option<&'static i32>);
    }
    size_and_align! {
        minicore: option;
        struct Goal(Option<Option<bool>>);
    }
}

#[test]
fn enums_with_discriminants() {
    size_and_align! {
        enum Goal {
            A = 1000,
            B = 2000,
            C = 3000,
        }
    }
    size_and_align! {
        enum Goal {
            A = 254,
            B,
            C, // implicitly becomes 256, so we need two bytes
        }
    }
}
