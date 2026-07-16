use query_group_macro::query_group;

mod logger_db;
use expect_test::expect;
use logger_db::LoggerDb;

#[salsa::input(singleton)]
struct InputString {
    inner: String,
}

#[query_group]
pub trait HelloWorldDatabase: salsa::Database {
    #[salsa::transparent]
    fn length_query(&self, key: ()) -> (usize, usize);
}

fn length_query(db: &dyn HelloWorldDatabase, _key: ()) -> (usize, usize) {
    let len = InputString::get(db).inner(db).len();
    (len, len)
}

#[test]
fn query() {
    let db = LoggerDb::default();

    _ = InputString::new(&db, String::from("Hello, world!"));
    let len = db.length_query(());

    assert_eq!(len, (13, 13));
    db.assert_logs(expect!["[]"]);
}
