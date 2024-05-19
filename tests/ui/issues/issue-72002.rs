//@ check-pass
struct Indexable;

impl Indexable {
    fn boo(&mut self) {}
}

impl std::ops::Index<&str> for Indexable {
    type Output = Indexable;

    fn index(&self, field: &str) -> &Indexable {
        self
    }
}

impl std::ops::IndexMut<&str> for Indexable {
    fn index_mut(&mut self, field: &str) -> &mut Indexable {
        self
    }
}

fn main() {
    let mut v = Indexable;
    let field = "hello".to_string();

    v[field.as_str()].boo();

    v[&field].boo(); // < This should work
}
