mod foo {
    pub struct B(pub ());
}

mod baz {
    fn foo() {
        B(());
        //~^ ERROR cannot find function, tuple struct or tuple variant `B`
    }
}

fn main() {}
