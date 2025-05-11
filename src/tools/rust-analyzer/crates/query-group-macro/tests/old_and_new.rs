use expect_test::expect;

mod logger_db;
use logger_db::LoggerDb;
use query_group_macro::query_group;

#[salsa_macros::input]
struct Input {
    str: String,
}

#[query_group]
trait PartialMigrationDatabase: salsa::Database {
    fn length_query(&self, input: Input) -> usize;

    // renamed/invoke query
    #[salsa::invoke(invoke_length_query_actual)]
    fn invoke_length_query(&self, input: Input) -> usize;

    // invoke tracked function
    #[salsa::invoke(invoke_length_tracked_actual)]
    fn invoke_length_tracked(&self, input: Input) -> usize;
}

fn length_query(db: &dyn PartialMigrationDatabase, input: Input) -> usize {
    input.str(db).len()
}

fn invoke_length_query_actual(db: &dyn PartialMigrationDatabase, input: Input) -> usize {
    input.str(db).len()
}

#[salsa_macros::tracked]
fn invoke_length_tracked_actual(db: &dyn PartialMigrationDatabase, input: Input) -> usize {
    input.str(db).len()
}

#[test]
fn unadorned_query() {
    let db = LoggerDb::default();

    let input = Input::new(&db, String::from("Hello, world!"));
    let len = db.length_query(input);

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: length_query_shim(Id(0)) })",
        ]"#]]);
}

#[test]
fn invoke_query() {
    let db = LoggerDb::default();

    let input = Input::new(&db, String::from("Hello, world!"));
    let len = db.invoke_length_query(input);

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: invoke_length_query_shim(Id(0)) })",
        ]"#]]);
}

// todo: does this even make sense?
#[test]
fn invoke_tracked_query() {
    let db = LoggerDb::default();

    let input = Input::new(&db, String::from("Hello, world!"));
    let len = db.invoke_length_tracked(input);

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: invoke_length_tracked_shim(Id(0)) })",
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: invoke_length_tracked_actual(Id(0)) })",
        ]"#]]);
}

#[test]
fn new_salsa_baseline() {
    let db = LoggerDb::default();

    #[salsa_macros::input]
    struct Input {
        str: String,
    }

    #[salsa_macros::tracked]
    fn new_salsa_length_query(db: &dyn PartialMigrationDatabase, input: Input) -> usize {
        input.str(db).len()
    }

    let input = Input::new(&db, String::from("Hello, world!"));
    let len = new_salsa_length_query(&db, input);

    assert_eq!(len, 13);
    db.assert_logs(expect![[r#"
        [
            "salsa_event(WillCheckCancellation)",
            "salsa_event(WillExecute { database_key: new_salsa_length_query(Id(0)) })",
        ]"#]]);
}
