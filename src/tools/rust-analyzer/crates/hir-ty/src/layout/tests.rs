use std::collections::HashMap;

use base_db::fixture::WithFixture;
use chalk_ir::{AdtId, TyKind};
use either::Either;
use hir_def::db::DefDatabase;
use triomphe::Arc;

use crate::{
    db::HirDatabase,
    layout::{Layout, LayoutError},
    test_db::TestDB,
    Interner, Substitution,
};

mod closure;

fn current_machine_data_layout() -> String {
    project_model::target_data_layout::get(None, None, &HashMap::default()).unwrap()
}

fn eval_goal(ra_fixture: &str, minicore: &str) -> Result<Arc<Layout>, LayoutError> {
    let target_data_layout = current_machine_data_layout();
    let ra_fixture = format!(
        "{minicore}//- /main.rs crate:test target_data_layout:{target_data_layout}\n{ra_fixture}",
    );

    let (db, file_ids) = TestDB::with_many_files(&ra_fixture);
    let adt_or_type_alias_id = file_ids
        .into_iter()
        .find_map(|file_id| {
            let module_id = db.module_for_file(file_id);
            let def_map = module_id.def_map(&db);
            let scope = &def_map[module_id.local_id].scope;
            let adt_or_type_alias_id = scope.declarations().find_map(|x| match x {
                hir_def::ModuleDefId::AdtId(x) => {
                    let name = match x {
                        hir_def::AdtId::StructId(x) => db.struct_data(x).name.to_smol_str(),
                        hir_def::AdtId::UnionId(x) => db.union_data(x).name.to_smol_str(),
                        hir_def::AdtId::EnumId(x) => db.enum_data(x).name.to_smol_str(),
                    };
                    (name == "Goal").then_some(Either::Left(x))
                }
                hir_def::ModuleDefId::TypeAliasId(x) => {
                    let name = db.type_alias_data(x).name.to_smol_str();
                    (name == "Goal").then_some(Either::Right(x))
                }
                _ => None,
            })?;
            Some(adt_or_type_alias_id)
        })
        .unwrap();
    let goal_ty = match adt_or_type_alias_id {
        Either::Left(adt_id) => {
            TyKind::Adt(AdtId(adt_id), Substitution::empty(Interner)).intern(Interner)
        }
        Either::Right(ty_id) => {
            db.ty(ty_id.into()).substitute(Interner, &Substitution::empty(Interner))
        }
    };
    db.layout_of_ty(
        goal_ty,
        db.trait_environment(match adt_or_type_alias_id {
            Either::Left(adt) => hir_def::GenericDefId::AdtId(adt),
            Either::Right(ty) => hir_def::GenericDefId::TypeAliasId(ty),
        }),
    )
}

/// A version of `eval_goal` for types that can not be expressed in ADTs, like closures and `impl Trait`
fn eval_expr(ra_fixture: &str, minicore: &str) -> Result<Arc<Layout>, LayoutError> {
    let target_data_layout = current_machine_data_layout();
    let ra_fixture = format!(
        "{minicore}//- /main.rs crate:test target_data_layout:{target_data_layout}\nfn main(){{let goal = {{{ra_fixture}}};}}",
    );

    let (db, file_id) = TestDB::with_single_file(&ra_fixture);
    let module_id = db.module_for_file(file_id);
    let def_map = module_id.def_map(&db);
    let scope = &def_map[module_id.local_id].scope;
    let function_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(x) => {
                let name = db.function_data(x).name.to_smol_str();
                (name == "main").then_some(x)
            }
            _ => None,
        })
        .unwrap();
    let hir_body = db.body(function_id.into());
    let b = hir_body.bindings.iter().find(|x| x.1.name.to_smol_str() == "goal").unwrap().0;
    let infer = db.infer(function_id.into());
    let goal_ty = infer.type_of_binding[b].clone();
    db.layout_of_ty(goal_ty, db.trait_environment(function_id.into()))
}

#[track_caller]
fn check_size_and_align(ra_fixture: &str, minicore: &str, size: u64, align: u64) {
    let l = eval_goal(ra_fixture, minicore).unwrap();
    assert_eq!(l.size.bytes(), size, "size mismatch");
    assert_eq!(l.align.abi.bytes(), align, "align mismatch");
}

#[track_caller]
fn check_size_and_align_expr(ra_fixture: &str, minicore: &str, size: u64, align: u64) {
    let l = eval_expr(ra_fixture, minicore).unwrap();
    assert_eq!(l.size.bytes(), size, "size mismatch");
    assert_eq!(l.align.abi.bytes(), align, "align mismatch");
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

#[macro_export]
macro_rules! size_and_align_expr {
    (minicore: $($x:tt),*; stmts: [$($s:tt)*] $($t:tt)*) => {
        {
            #[allow(dead_code)]
            #[allow(unused_must_use)]
            #[allow(path_statements)]
            {
                $($s)*
                let val = { $($t)* };
                $crate::layout::tests::check_size_and_align_expr(
                    &format!("{{ {} let val = {{ {} }}; val }}", stringify!($($s)*), stringify!($($t)*)),
                    &format!("//- minicore: {}\n", stringify!($($x),*)),
                    ::std::mem::size_of_val(&val) as u64,
                    ::std::mem::align_of_val(&val) as u64,
                );
            }
        }
    };
    ($($t:tt)*) => {
        {
            #[allow(dead_code)]
            {
                let val = { $($t)* };
                $crate::layout::tests::check_size_and_align_expr(
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
fn associated_types() {
    size_and_align! {
        trait Tr {
            type Ty;
        }

        impl Tr for i32 {
            type Ty = i64;
        }

        struct Foo<A: Tr>(<A as Tr>::Ty);
        struct Bar<A: Tr>(A::Ty);
        struct Goal(Foo<i32>, Bar<i32>, <i32 as Tr>::Ty);
    }
    check_size_and_align(
        r#"
//- /b/mod.rs crate:b
pub trait Tr {
    type Ty;
}
pub struct Foo<A: Tr>(<A as Tr>::Ty);

//- /a/mod.rs crate:a deps:b
use b::{Tr, Foo};

struct S;
impl Tr for S {
    type Ty = i64;
}
struct Goal(Foo<S>);
        "#,
        "",
        8,
        8,
    );
}

#[test]
fn simd_types() {
    check_size_and_align(
        r#"
            #[repr(simd)]
            struct SimdType(i64, i64);
            struct Goal(SimdType);
        "#,
        "",
        16,
        16,
    );
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
        minicore: iterators;
        stmts: []
        trait Tr {}
        impl Tr for i32 {}
        fn foo() -> impl Iterator<Item = impl Tr> {
            [1, 2, 3].into_iter()
        }
        let mut iter = foo();
        let item = iter.next();
        (iter, item)
    }
    size_and_align_expr! {
        minicore: future;
        stmts: []
        use core::{future::Future, task::{Poll, Context}, pin::pin};
        use std::{task::Wake, sync::Arc};
        trait Tr {}
        impl Tr for i32 {}
        async fn f() -> impl Tr {
            2
        }
        fn unwrap_fut<T>(inp: impl Future<Output = T>) -> Poll<T> {
            // In a normal test we could use `loop {}` or `panic!()` here,
            // but rustc actually runs this code.
            let pinned = pin!(inp);
            struct EmptyWaker;
            impl Wake for EmptyWaker {
                fn wake(self: Arc<Self>) {
                }
            }
            let waker = Arc::new(EmptyWaker).into();
            let mut context = Context::from_waker(&waker);
            let x = pinned.poll(&mut context);
            x
        }
        let x = unwrap_fut(f());
        x
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
fn unsized_ref() {
    size_and_align! {
        struct S1([u8]);
        struct S2(S1);
        struct S3(i32, str);
        struct S4(u64, S3);
        #[allow(dead_code)]
        struct S5 {
            field1: u8,
            field2: i16,
            field_last: S4,
        }

        struct Goal(&'static S1, &'static S2, &'static S3, &'static S4, &'static S5);
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
fn non_zero_and_non_null() {
    size_and_align! {
        minicore: non_zero, non_null, option;
        use core::{num::NonZeroU8, ptr::NonNull};
        struct Goal(Option<NonZeroU8>, Option<NonNull<i32>>);
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
fn const_eval() {
    size_and_align! {
        struct Goal([i32; 2 + 2]);
    }
    size_and_align! {
        const X: usize = 5;
        struct Goal([i32; X]);
    }
    size_and_align! {
        mod foo {
            pub(super) const BAR: usize = 5;
        }
        struct Ar<T>([T; foo::BAR]);
        struct Goal(Ar<Ar<i32>>);
    }
    size_and_align! {
        type Goal = [u8; 2 + 2];
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
    size_and_align! {
        enum Goal {
            A = 1, // This one is (perhaps surprisingly) zero sized.
        }
    }
}

#[test]
fn core_mem_discriminant() {
    size_and_align! {
        minicore: discriminant;
        struct S(i32, u64);
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        #[repr(u32)]
        enum S {
            A,
            B,
            C,
        }
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        enum S {
            A(i32),
            B(i64),
            C(u8),
        }
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        #[repr(C, u16)]
        enum S {
            A(i32),
            B(i64) = 200,
            C = 1000,
        }
        struct Goal(core::mem::Discriminant<S>);
    }
}
