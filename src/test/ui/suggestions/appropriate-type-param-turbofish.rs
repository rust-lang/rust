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

fn foo() {
    x(); //~ ERROR type annotations needed
}

fn bar() {
    let _ = x(); //~ ERROR type annotations needed
}

fn main() {}
