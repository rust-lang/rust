//! Test that transparent (uncached) queries work

#[salsa::query_group(QueryGroupStorage)]
trait QueryGroup {
    #[salsa::input]
    fn input(&self, x: u32) -> u32;
    #[salsa::transparent]
    fn wrap(&self, x: u32) -> u32;
    fn get(&self, x: u32) -> u32;
}

fn wrap(db: &dyn QueryGroup, x: u32) -> u32 {
    db.input(x)
}

fn get(db: &dyn QueryGroup, x: u32) -> u32 {
    db.wrap(x)
}

#[salsa::database(QueryGroupStorage)]
#[derive(Default)]
struct Database {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for Database {}

#[test]
fn transparent_queries_work() {
    let mut db = Database::default();

    db.set_input(1, 10);
    assert_eq!(db.get(1), 10);
    assert_eq!(db.get(1), 10);

    db.set_input(1, 92);
    assert_eq!(db.get(1), 92);
    assert_eq!(db.get(1), 92);
}
