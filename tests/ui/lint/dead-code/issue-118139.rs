// run-pass

enum Category {
    A,
    B,
}

trait Foo {
    fn foo(&self) -> Category {
        Category::A
    }
}

fn main() {
    let _c = Category::B;
}
