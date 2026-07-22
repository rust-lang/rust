mod logger_db;
use expect_test::expect;
use logger_db::LoggerDb;

use query_group_macro::query_group;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error;

#[salsa::input(singleton)]
struct InputString {
    inner: String,
}

#[query_group]
pub trait ResultDatabase: salsa::Database {
    #[salsa::transparent]
    fn length(&self, key: ()) -> Result<usize, Error>;

    #[salsa::transparent]
    fn length2(&self, key: ()) -> Result<usize, Error>;
}

fn length(db: &dyn ResultDatabase, _key: ()) -> Result<usize, Error> {
    Ok(InputString::get(db).inner(db).len())
}

fn length2(db: &dyn ResultDatabase, _key: ()) -> Result<usize, Error> {
    Ok(InputString::get(db).inner(db).len())
}

#[test]
fn test_queries_with_results() {
    let db = LoggerDb::default();
    let input = "hello";
    _ = InputString::new(&db, input.to_owned());
    assert_eq!(db.length(()), Ok(input.len()));
    assert_eq!(db.length2(()), Ok(input.len()));

    db.assert_logs(expect!["[]"]);
}
