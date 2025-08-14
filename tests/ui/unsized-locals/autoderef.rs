#![feature(unsized_fn_params)]

pub trait Foo {
    fn foo(self) -> String;
}

impl Foo for [char] {
    fn foo(self) -> String {
        self.iter().collect()
    }
}

impl Foo for str {
    fn foo(self) -> String {
        self.to_owned()
    }
}

impl Foo for dyn FnMut() -> String {
    fn foo(mut self) -> String {
        self()
    }
}

fn main() {
    let x = *(Box::new(['h', 'e', 'l', 'l', 'o']) as Box<[char]>); //~ERROR the size for values of type `[char]` cannot be known at compilation time
    assert_eq!(&x.foo() as &str, "hello");

    let x = Box::new(['h', 'e', 'l', 'l', 'o']) as Box<[char]>;
    assert_eq!(&x.foo() as &str, "hello");

    let x = "hello".to_owned().into_boxed_str();
    assert_eq!(&x.foo() as &str, "hello");

    let x = *("hello".to_owned().into_boxed_str()); //~ERROR the size for values of type `str` cannot be known at compilation time
    assert_eq!(&x.foo() as &str, "hello");

    let x = "hello".to_owned().into_boxed_str();
    assert_eq!(&x.foo() as &str, "hello");

    let x = *(Box::new(|| "hello".to_owned()) as Box<dyn FnMut() -> String>); //~ERROR the size for values of type `dyn FnMut() -> String` cannot be known at compilation time
    assert_eq!(&x.foo() as &str, "hello");

    let x = Box::new(|| "hello".to_owned()) as Box<dyn FnMut() -> String>;
    assert_eq!(&x.foo() as &str, "hello");
}
