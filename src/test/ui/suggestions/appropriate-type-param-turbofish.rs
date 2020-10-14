mod a {
    fn foo() {
        vec![1, 2, 3].into_iter().collect(); //~ ERROR type annotations needed
    }
    fn bar() {
        vec!["a", "b", "c"].into_iter().collect(); //~ ERROR type annotations needed
    }
    fn qux() {
        vec!['a', 'b', 'c'].into_iter().collect(); //~ ERROR type annotations needed
    }
}
mod b {
    fn foo() {
        let _ = vec![1, 2, 3].into_iter().collect(); //~ ERROR type annotations needed
    }
    fn bar() {
        let _ = vec!["a", "b", "c"].into_iter().collect(); //~ ERROR type annotations needed
    }
    fn qux() {
        let _ = vec!['a', 'b', 'c'].into_iter().collect(); //~ ERROR type annotations needed
    }
}

mod c {
    fn foo() {
        let _x = vec![1, 2, 3].into_iter().collect(); //~ ERROR type annotations needed
    }
    fn bar() {
        let _x = vec!["a", "b", "c"].into_iter().collect(); //~ ERROR type annotations needed
    }
    fn qux() {
        let _x = vec!['a', 'b', 'c'].into_iter().collect(); //~ ERROR type annotations needed
    }
}

trait T: Sized {
    fn new() -> Self;
}

fn x<X: T>() -> X {
    T::new()
}
struct S;
impl T for S {
    fn new() -> Self {
        S
    }
}

struct AAA<X> { x: X, }

impl<Z: std::fmt::Display> std::iter::FromIterator<Z> for AAA<Z> {
    fn from_iter<I: IntoIterator<Item = Z>>(_iter: I) -> AAA<Z> {
        panic!()
    }
}

fn foo() {
    x(); //~ ERROR type annotations needed
}

fn bar() {
    let _ = x(); //~ ERROR type annotations needed
}

fn qux() {
    let x: Vec<&std::path::Path> = vec![];
    let y = x.into_iter().collect(); //~ ERROR type annotations needed
}

trait Foo {
    fn foo<T: Bar<K>, K>(&self) -> T {
        panic!()
    }
}

trait Bar<X> {}

struct R<X>(X);
impl<X> Bar<X> for R<X> {}
struct K<X>(X);
impl Bar<i32> for K<i32> {}

struct I;

impl Foo for I {}

fn bat() {
    let _ = I.foo(); //~ ERROR type annotations needed
}

// This case does not have a good suggestion yet:
fn bak<T: Into<String>>() {}

fn main() {
    bak(); //~ ERROR type annotations needed
}
