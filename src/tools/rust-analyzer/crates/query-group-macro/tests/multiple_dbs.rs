use query_group_macro::query_group;

#[query_group]
pub trait DatabaseOne: salsa::Database {
    #[salsa::input]
    fn input_string(&self) -> String;

    // unadorned query
    #[salsa::invoke_interned(length)]
    fn length(&self, key: ()) -> usize;
}

#[query_group]
pub trait DatabaseTwo: DatabaseOne {
    #[salsa::invoke_interned(second_length)]
    fn second_length(&self, key: ()) -> usize;
}

fn length(db: &dyn DatabaseOne, _key: ()) -> usize {
    db.input_string().len()
}

fn second_length(db: &dyn DatabaseTwo, _key: ()) -> usize {
    db.input_string().len()
}
