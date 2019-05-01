// Test that we properly detect the cycle amongst the traits
// here and report an error.

use std::panic::RefUnwindSafe;

trait Database {
    type Storage;
}
trait HasQueryGroup {}
trait Query<DB> {
    type Data;
}
trait SourceDatabase {
    fn parse(&self) {
        loop {}
    }
}

struct ParseQuery;
struct RootDatabase {
    _runtime: Runtime<RootDatabase>,
}
struct Runtime<DB: Database> {
    _storage: Box<DB::Storage>,
}
struct SalsaStorage {
    _parse: <ParseQuery as Query<RootDatabase>>::Data, //~ ERROR overflow
}

impl Database for RootDatabase { //~ ERROR overflow
    type Storage = SalsaStorage;
}
impl HasQueryGroup for RootDatabase {}
impl<DB> Query<DB> for ParseQuery
where
    DB: SourceDatabase,
    DB: Database,
{
    type Data = RootDatabase;
}
impl<T> SourceDatabase for T
where
    T: RefUnwindSafe,
    T: HasQueryGroup,
{
}

pub(crate) fn goto_implementation(db: &RootDatabase) -> u32 {
    // This is not satisfied:
    //
    // - `RootDatabase: SourceDatabase`
    //   - requires `RootDatabase: RefUnwindSafe` + `RootDatabase: HasQueryGroup`
    // - `RootDatabase: RefUnwindSafe`
    //   - requires `Runtime<RootDatabase>: RefUnwindSafe`
    // - `Runtime<RootDatabase>: RefUnwindSafe`
    //   - requires `DB::Storage: RefUnwindSafe` (`SalsaStorage: RefUnwindSafe`)
    // - `SalsaStorage: RefUnwindSafe`
    //    - requires `<ParseQuery as Query<RootDatabase>>::Data: RefUnwindSafe`,
    //      which means `ParseQuery: Query<RootDatabase>`
    // - `ParseQuery: Query<RootDatabase>`
    //    - requires `RootDatabase: SourceDatabase`,
    // - `RootDatabase: SourceDatabase` is already on the stack, so we have a
    //   cycle with non-coinductive participants
    //
    // we used to fail to report an error here because we got the
    // caching wrong.
    SourceDatabase::parse(db);
    22
}

fn main() {}
