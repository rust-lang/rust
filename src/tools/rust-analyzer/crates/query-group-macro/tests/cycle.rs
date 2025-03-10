use std::panic::UnwindSafe;

use expect_test::expect;
use query_group_macro::query_group;
use salsa::Setter;

/// The queries A, B, and C in `Database` can be configured
/// to invoke one another in arbitrary ways using this
/// enum.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum CycleQuery {
    None,
    A,
    B,
    C,
    AthenC,
}

#[salsa::input]
struct ABC {
    a: CycleQuery,
    b: CycleQuery,
    c: CycleQuery,
}

impl CycleQuery {
    fn invoke(self, db: &dyn CycleDatabase, abc: ABC) -> Result<(), Error> {
        match self {
            CycleQuery::A => db.cycle_a(abc),
            CycleQuery::B => db.cycle_b(abc),
            CycleQuery::C => db.cycle_c(abc),
            CycleQuery::AthenC => {
                let _ = db.cycle_a(abc);
                db.cycle_c(abc)
            }
            CycleQuery::None => Ok(()),
        }
    }
}

#[salsa::input]
struct MyInput {}

#[salsa::tracked]
fn memoized_a(db: &dyn CycleDatabase, input: MyInput) {
    memoized_b(db, input)
}

#[salsa::tracked]
fn memoized_b(db: &dyn CycleDatabase, input: MyInput) {
    memoized_a(db, input)
}

#[salsa::tracked]
fn volatile_a(db: &dyn CycleDatabase, input: MyInput) {
    db.report_untracked_read();
    volatile_b(db, input)
}

#[salsa::tracked]
fn volatile_b(db: &dyn CycleDatabase, input: MyInput) {
    db.report_untracked_read();
    volatile_a(db, input)
}

#[track_caller]
fn extract_cycle(f: impl FnOnce() + UnwindSafe) -> salsa::Cycle {
    let v = std::panic::catch_unwind(f);
    if let Err(d) = &v {
        if let Some(cycle) = d.downcast_ref::<salsa::Cycle>() {
            return cycle.clone();
        }
    }
    panic!("unexpected value: {:?}", v)
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct Error {
    cycle: Vec<String>,
}

#[query_group]
trait CycleDatabase: salsa::Database {
    #[salsa::cycle(recover_a)]
    fn cycle_a(&self, abc: ABC) -> Result<(), Error>;

    #[salsa::cycle(recover_b)]
    fn cycle_b(&self, abc: ABC) -> Result<(), Error>;

    fn cycle_c(&self, abc: ABC) -> Result<(), Error>;
}

fn cycle_a(db: &dyn CycleDatabase, abc: ABC) -> Result<(), Error> {
    abc.a(db).invoke(db, abc)
}

fn recover_a(
    _db: &dyn CycleDatabase,
    cycle: &salsa::Cycle,
    _: CycleDatabaseData,
    _abc: ABC,
) -> Result<(), Error> {
    Err(Error { cycle: cycle.participant_keys().map(|k| format!("{k:?}")).collect() })
}

fn cycle_b(db: &dyn CycleDatabase, abc: ABC) -> Result<(), Error> {
    abc.b(db).invoke(db, abc)
}

fn recover_b(
    _db: &dyn CycleDatabase,
    cycle: &salsa::Cycle,
    _: CycleDatabaseData,
    _abc: ABC,
) -> Result<(), Error> {
    Err(Error { cycle: cycle.participant_keys().map(|k| format!("{k:?}")).collect() })
}

fn cycle_c(db: &dyn CycleDatabase, abc: ABC) -> Result<(), Error> {
    abc.c(db).invoke(db, abc)
}

#[test]
fn cycle_memoized() {
    let db = salsa::DatabaseImpl::new();

    let input = MyInput::new(&db);
    let cycle = extract_cycle(|| memoized_a(&db, input));
    let expected = expect![[r#"
        [
            DatabaseKeyIndex(
                IngredientIndex(
                    1,
                ),
                Id(0),
            ),
            DatabaseKeyIndex(
                IngredientIndex(
                    2,
                ),
                Id(0),
            ),
        ]
    "#]];
    expected.assert_debug_eq(&cycle.all_participants(&db));
}

#[test]
fn inner_cycle() {
    //     A --> B <-- C
    //     ^     |
    //     +-----+
    let db = salsa::DatabaseImpl::new();

    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::A, CycleQuery::B);
    let err = db.cycle_c(abc);
    assert!(err.is_err());
    let expected = expect![[r#"
            [
                "cycle_a_shim(Id(1400))",
                "cycle_b_shim(Id(1000))",
            ]
        "#]];
    expected.assert_debug_eq(&err.unwrap_err().cycle);
}

#[test]
fn cycle_revalidate() {
    //     A --> B
    //     ^     |
    //     +-----+
    let mut db = salsa::DatabaseImpl::new();
    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::A, CycleQuery::None);
    assert!(db.cycle_a(abc).is_err());
    abc.set_b(&mut db).to(CycleQuery::A); // same value as default
    assert!(db.cycle_a(abc).is_err());
}

#[test]
fn cycle_recovery_unchanged_twice() {
    //     A --> B
    //     ^     |
    //     +-----+
    let mut db = salsa::DatabaseImpl::new();
    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::A, CycleQuery::None);
    assert!(db.cycle_a(abc).is_err());

    abc.set_c(&mut db).to(CycleQuery::A); // force new revision
    assert!(db.cycle_a(abc).is_err());
}

#[test]
fn cycle_appears() {
    let mut db = salsa::DatabaseImpl::new();
    //     A --> B
    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::None, CycleQuery::None);
    assert!(db.cycle_a(abc).is_ok());

    //     A --> B
    //     ^     |
    //     +-----+
    abc.set_b(&mut db).to(CycleQuery::A);
    assert!(db.cycle_a(abc).is_err());
}

#[test]
fn cycle_disappears() {
    let mut db = salsa::DatabaseImpl::new();

    //     A --> B
    //     ^     |
    //     +-----+
    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::A, CycleQuery::None);
    assert!(db.cycle_a(abc).is_err());

    //     A --> B
    abc.set_b(&mut db).to(CycleQuery::None);
    assert!(db.cycle_a(abc).is_ok());
}

#[test]
fn cycle_multiple() {
    // No matter whether we start from A or B, we get the same set of participants:
    let db = salsa::DatabaseImpl::new();

    // Configuration:
    //
    //     A --> B <-- C
    //     ^     |     ^
    //     +-----+     |
    //           |     |
    //           +-----+
    //
    // Here, conceptually, B encounters a cycle with A and then
    // recovers.
    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::AthenC, CycleQuery::A);

    let c = db.cycle_c(abc);
    let b = db.cycle_b(abc);
    let a = db.cycle_a(abc);
    let expected = expect![[r#"
        (
            [
                "cycle_a_shim(Id(1000))",
                "cycle_b_shim(Id(1400))",
            ],
            [
                "cycle_a_shim(Id(1000))",
                "cycle_b_shim(Id(1400))",
            ],
            [
                "cycle_a_shim(Id(1000))",
                "cycle_b_shim(Id(1400))",
            ],
        )
    "#]];
    expected.assert_debug_eq(&(c.unwrap_err().cycle, b.unwrap_err().cycle, a.unwrap_err().cycle));
}

#[test]
fn cycle_mixed_1() {
    let db = salsa::DatabaseImpl::new();
    //     A --> B <-- C
    //           |     ^
    //           +-----+
    let abc = ABC::new(&db, CycleQuery::B, CycleQuery::C, CycleQuery::B);

    let expected = expect![[r#"
        [
            "cycle_b_shim(Id(1000))",
            "cycle_c_shim(Id(c00))",
        ]
    "#]];
    expected.assert_debug_eq(&db.cycle_c(abc).unwrap_err().cycle);
}
