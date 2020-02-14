trait NotObjectSafe {
    fn foo() -> Self;
}

trait ObjectSafe {
    fn bar(&self);
}

struct A;
struct B;

impl NotObjectSafe for A {
    fn foo() -> Self {
        A
    }
}

impl NotObjectSafe for B {
    fn foo() -> Self {
        B
    }
}

impl ObjectSafe for A {
    fn bar(&self) {}
}

impl ObjectSafe for B {
    fn bar(&self) {}
}

fn can() -> impl NotObjectSafe {
    if true {
        return A;
    }
    B //~ ERROR mismatched types
}

fn cat() -> impl ObjectSafe {
    if true {
        return A;
    }
    B //~ ERROR mismatched types
}

fn main() {}
