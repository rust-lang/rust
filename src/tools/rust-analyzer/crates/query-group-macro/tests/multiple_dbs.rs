use query_group_macro::query_group;

#[salsa::input(singleton)]
struct InputString {
    inner: String,
}

#[query_group]
pub trait DatabaseOne: salsa::Database {
    // unadorned query
    #[salsa::transparent]
    fn length(&self, key: ()) -> usize;
}

#[query_group]
pub trait DatabaseTwo: DatabaseOne {
    #[salsa::transparent]
    fn second_length(&self, key: ()) -> usize;
}

fn length(db: &dyn DatabaseOne, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}

fn second_length(db: &dyn DatabaseTwo, _key: ()) -> usize {
    InputString::get(db).inner(db).len()
}
