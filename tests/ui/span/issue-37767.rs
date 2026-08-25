trait A {
    fn foo(&mut self) {}
}

trait B : A {
    fn foo(&mut self) {}
}

fn bar<T: B>(a: &T) {
    a.foo() //~ ERROR multiple applicable items
}

trait C {
    fn foo(&self) {}
}

trait D : C {
    fn foo(&self) {}
}

fn quz<T: D>(a: &T) {
    a.foo() //~ ERROR multiple applicable items
}

trait E : Sized {
    fn foo(self) {}
}

trait F : E {
    fn foo(self) {}
}

fn foo<T: F>(a: T) {
    a.foo() //~ ERROR multiple applicable items
}

fn pass<T: C>(a: &T) {
    a.foo()
}

fn main() {}
