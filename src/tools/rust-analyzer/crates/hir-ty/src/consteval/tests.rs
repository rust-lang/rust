use base_db::fixture::WithFixture;
use hir_def::{db::DefDatabase, expr::Literal};

use crate::{consteval::ComputedExpr, db::HirDatabase, test_db::TestDB};

use super::ConstEvalError;

fn check_fail(ra_fixture: &str, error: ConstEvalError) {
    assert_eq!(eval_goal(ra_fixture), Err(error));
}

fn check_number(ra_fixture: &str, answer: i128) {
    let r = eval_goal(ra_fixture).unwrap();
    match r {
        ComputedExpr::Literal(Literal::Int(r, _)) => assert_eq!(r, answer),
        ComputedExpr::Literal(Literal::Uint(r, _)) => assert_eq!(r, answer as u128),
        x => panic!("Expected number but found {:?}", x),
    }
}

fn eval_goal(ra_fixture: &str) -> Result<ComputedExpr, ConstEvalError> {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let module_id = db.module_for_file(file_id);
    let def_map = module_id.def_map(&db);
    let scope = &def_map[module_id.local_id].scope;
    let const_id = scope
        .declarations()
        .into_iter()
        .find_map(|x| match x {
            hir_def::ModuleDefId::ConstId(x) => {
                if db.const_data(x).name.as_ref()?.to_string() == "GOAL" {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .unwrap();
    db.const_eval(const_id)
}

#[test]
fn add() {
    check_number(r#"const GOAL: usize = 2 + 2;"#, 4);
}

#[test]
fn bit_op() {
    check_number(r#"const GOAL: u8 = !0 & !(!0 >> 1)"#, 128);
    check_number(r#"const GOAL: i8 = !0 & !(!0 >> 1)"#, 0);
    // FIXME: rustc evaluate this to -128
    check_fail(
        r#"const GOAL: i8 = 1 << 7"#,
        ConstEvalError::Panic("attempt to run invalid arithmetic operation".to_string()),
    );
    check_fail(
        r#"const GOAL: i8 = 1 << 8"#,
        ConstEvalError::Panic("attempt to run invalid arithmetic operation".to_string()),
    );
}

#[test]
fn locals() {
    check_number(
        r#"
    const GOAL: usize = {
        let a = 3 + 2;
        let b = a * a;
        b
    };
    "#,
        25,
    );
}

#[test]
fn consts() {
    check_number(
        r#"
    const F1: i32 = 1;
    const F3: i32 = 3 * F2;
    const F2: i32 = 2 * F1;
    const GOAL: i32 = F3;
    "#,
        6,
    );
}

#[test]
fn enums() {
    check_number(
        r#"
    enum E {
        F1 = 1,
        F2 = 2 * E::F1 as u8,
        F3 = 3 * E::F2 as u8,
    }
    const GOAL: i32 = E::F3 as u8;
    "#,
        6,
    );
    check_number(
        r#"
    enum E { F1 = 1, F2, }
    const GOAL: i32 = E::F2 as u8;
    "#,
        2,
    );
    check_number(
        r#"
    enum E { F1, }
    const GOAL: i32 = E::F1 as u8;
    "#,
        0,
    );
    let r = eval_goal(
        r#"
        enum E { A = 1, }
        const GOAL: E = E::A;
        "#,
    )
    .unwrap();
    match r {
        ComputedExpr::Enum(name, _, Literal::Uint(val, _)) => {
            assert_eq!(name, "E::A");
            assert_eq!(val, 1);
        }
        x => panic!("Expected enum but found {:?}", x),
    }
}

#[test]
fn const_loop() {
    check_fail(
        r#"
    const F1: i32 = 1 * F3;
    const F3: i32 = 3 * F2;
    const F2: i32 = 2 * F1;
    const GOAL: i32 = F3;
    "#,
        ConstEvalError::Loop,
    );
}

#[test]
fn const_impl_assoc() {
    check_number(
        r#"
    struct U5;
    impl U5 {
        const VAL: usize = 5;
    }
    const GOAL: usize = U5::VAL;
    "#,
        5,
    );
}

#[test]
fn const_generic_subst() {
    // FIXME: this should evaluate to 5
    check_fail(
        r#"
    struct Adder<const N: usize, const M: usize>;
    impl<const N: usize, const M: usize> Adder<N, M> {
        const VAL: usize = N + M;
    }
    const GOAL: usize = Adder::<2, 3>::VAL;
    "#,
        ConstEvalError::NotSupported("const generic without substitution"),
    );
}

#[test]
fn const_trait_assoc() {
    // FIXME: this should evaluate to 0
    check_fail(
        r#"
    struct U0;
    trait ToConst {
        const VAL: usize;
    }
    impl ToConst for U0 {
        const VAL: usize = 0;
    }
    const GOAL: usize = U0::VAL;
    "#,
        ConstEvalError::IncompleteExpr,
    );
}
