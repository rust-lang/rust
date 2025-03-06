use query_group_macro::query_group;

#[query_group]
pub trait DatabaseOne: salsa::Database {
    #[salsa::input]
    fn input_string(&self) -> String;

    // unadorned query
    fn length(&self, key: ()) -> usize;
}

#[query_group]
pub trait DatabaseTwo: DatabaseOne {
    fn second_length(&self, key: ()) -> usize;
}

fn length(db: &dyn DatabaseOne, _key: ()) -> usize {
    db.input_string().len()
}

fn second_length(db: &dyn DatabaseTwo, _key: ()) -> usize {
    db.input_string().len()
}
