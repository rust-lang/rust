trait B {
    fn foo(mut a: &String) {
        a.push_str("bar"); //~ ERROR cannot borrow immutable borrowed content
    }
}

pub fn foo<'a>(mut a: &'a String) {
    a.push_str("foo"); //~ ERROR cannot borrow immutable borrowed content
}

struct A {}

impl A {
    pub fn foo(mut a: &String) {
        a.push_str("foo"); //~ ERROR cannot borrow immutable borrowed content
    }
}

fn main() {
    foo(&"a".to_string());
    A::foo(&"a".to_string());
}
