use foo::bar;

mod foo {
    pub fn bar() {

    }
}

mod baz {
    pub fn bar() {

    }
}

fn main() {
    bar();
    baz::bar();
}
