mod foo {
    pub struct B(());
}

mod bar {
    use foo::B;

    fn foo() {
        B(()); //~ ERROR expected function, found struct `B` [E0423]
    }
}

mod baz {
    fn foo() {
        B(()); //~ ERROR cannot find function `B` in this scope [E0425]
    }
}

fn main() {}
