#[salsa::query_group(HelloWorld)]
trait HelloWorldDatabase: salsa::Database {
    #[salsa::input]
    fn input(&self, a: u32, b: u32) -> u32;

    fn none(&self) -> u32;

    fn one(&self, k: u32) -> u32;

    fn two(&self, a: u32, b: u32) -> u32;

    fn trailing(&self, a: u32, b: u32) -> u32;
}

fn none(_db: &dyn HelloWorldDatabase) -> u32 {
    22
}

fn one(_db: &dyn HelloWorldDatabase, k: u32) -> u32 {
    k * 2
}

fn two(_db: &dyn HelloWorldDatabase, a: u32, b: u32) -> u32 {
    a * b
}

fn trailing(_db: &dyn HelloWorldDatabase, a: u32, b: u32) -> u32 {
    a - b
}

#[salsa::database(HelloWorld)]
#[derive(Default)]
struct DatabaseStruct {
    storage: salsa::Storage<Self>,
}

impl salsa::Database for DatabaseStruct {}

#[test]
fn execute() {
    let mut db = DatabaseStruct::default();

    // test what happens with inputs:
    db.set_input(1, 2, 3);
    assert_eq!(db.input(1, 2), 3);

    assert_eq!(db.none(), 22);
    assert_eq!(db.one(11), 22);
    assert_eq!(db.two(11, 2), 22);
    assert_eq!(db.trailing(24, 2), 22);
}
