#[salsa::query_group(MyStruct)]
trait MyDatabase: salsa::Database {
    #[salsa::invoke(another_module::another_name)]
    fn my_query(&self, key: ()) -> ();
}

mod another_module {
    pub(crate) fn another_name(_: &dyn crate::MyDatabase, (): ()) {}
}

fn main() {}
