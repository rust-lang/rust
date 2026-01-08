// Test for issue #150320: suggest field reordering when a struct field
// moves a value and another field tries to borrow from it.

struct Foo {
    a: String,
    b: usize,
}

impl Foo {
    fn new(a: String) -> Self {
        Self {
            a,
            b: a.len(), //~ ERROR borrow of moved value: `a`
        }
    }
}

fn main() {
    let _ = Foo::new("hello".to_string());
}
