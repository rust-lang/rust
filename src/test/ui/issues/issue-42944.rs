mod foo {
    pub struct Bx(());
}

mod bar {
    use foo::Bx;

    fn foo() {
        Bx(());
        //~^ ERROR expected function, tuple struct or tuple variant, found struct `Bx` [E0423]
    }
}

mod baz {
    fn foo() {
        Bx(());
        //~^ ERROR cannot find function, tuple struct or tuple variant `Bx` in this scope [E0425]
    }
}

fn main() {}
