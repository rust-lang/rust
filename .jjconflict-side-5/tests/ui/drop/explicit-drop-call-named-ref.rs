//@ run-rustfix

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn foo(f: Foo) {
    Drop::drop(&mut f); //~ ERROR explicit use of destructor method
    //~| HELP: consider using `drop` function
}

fn main() {
    let f = Foo;
    foo(f);
}
