// A regression test for #111729 checking that we correctly
// track recursion depth for obligations returned by confirmation.
use std::panic::RefUnwindSafe;

trait Database {
    type Storage;
}
trait Query<DB> {
    type Data;
}
struct ParseQuery;
struct RootDatabase {
    _runtime: Runtime<RootDatabase>,
}

impl<T: RefUnwindSafe> Database for T {
    type Storage = SalsaStorage;
}
impl Database for RootDatabase {
    //~^ ERROR conflicting implementations of trait `Database` for type `RootDatabase`
    type Storage = SalsaStorage;
}

struct Runtime<DB: Database> {
    _storage: Box<DB::Storage>,
}
struct SalsaStorage {
    _parse: <ParseQuery as Query<RootDatabase>>::Data,
}

impl<DB: Database> Query<DB> for ParseQuery {
    type Data = RootDatabase;
}
fn main() {}
