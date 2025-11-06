trait A {
    fn foo(&mut self) {}
}

trait B {
    fn foo(&mut self) {}
}

fn bar<T: A + B>(a: &T) {
    a.foo() //~ ERROR multiple applicable items
}

trait C {
    fn foo(&self) {}
}

trait D {
    fn foo(&self) {}
}

fn quz<T: C + D>(a: &T) {
    a.foo() //~ ERROR multiple applicable items
}

trait E: Sized {
    fn foo(self) {}
}

trait F: Sized {
    fn foo(self) {}
}

fn foo<T: E + F>(a: T) {
    a.foo() //~ ERROR multiple applicable items
}

fn pass<T: C>(a: &T) {
    a.foo()
}

fn main() {}
