mod foo {
    pub struct Bx(pub(in crate::foo) ());
}

mod bar {
    use crate::foo::Bx;

    fn foo() {
        Bx(());
        //~^ ERROR cannot initialize a tuple struct which contains private fields [E0423]
    }
}

mod baz {
    fn foo() {
        Bx(());
        //~^ ERROR cannot find function, tuple struct or tuple variant `Bx` in this scope [E0425]
    }
}

fn main() {}
