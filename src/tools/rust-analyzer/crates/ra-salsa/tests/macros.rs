#[ra_salsa::query_group(MyStruct)]
trait MyDatabase: ra_salsa::Database {
    #[ra_salsa::invoke(another_module::another_name)]
    fn my_query(&self, key: ()) -> ();
}

mod another_module {
    #[allow(dead_code)]
    pub(crate) fn another_name(_: &dyn crate::MyDatabase, (): ()) {}
}

fn main() {}
