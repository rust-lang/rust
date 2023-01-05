trait Foo {
    fn foo(&self);
}

trait Bar {}

fn do_stuff<T : Bar>(t : T) {
    t.foo() //~ ERROR no method named `foo` found
}

fn main() {}
