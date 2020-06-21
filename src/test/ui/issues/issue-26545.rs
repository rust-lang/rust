mod foo {
    pub struct B(pub ());
}

mod baz {
    fn foo() {
        B(());
        //~^ ERROR cannot find function, tuple struct or tuple variant `B` in this scope [E0425]
    }
}

fn main() {}
