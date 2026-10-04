//! Checks borrow after move error when using `self` consuming method with struct update syntax.

struct Mine {
    test: String,
    other_val: isize,
}

impl Mine {
    fn make_string_bar(mut self) -> Mine {
        self.test = "Bar".to_string();
        self
    }
}

fn main() {
    let start = Mine { test: "Foo".to_string(), other_val: 0 };
    let end = Mine { other_val: 1, ..start.make_string_bar() };
    println!("{}", start.test); //~ ERROR borrow of moved value: `start`
}
