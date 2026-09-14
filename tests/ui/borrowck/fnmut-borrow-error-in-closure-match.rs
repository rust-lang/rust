//! Regression test for https://github.com/rust-lang/rust/issues/28971
enum Foo {
    Bar(u8)
}
fn main(){
    foo(|| {
        match Foo::Bar(1) {
            Foo::Baz(..) => (),
            //~^ ERROR no variant, associated function, or constant named `Baz` found
            _ => (),
        }
    });
}

fn foo<F>(f: F) where F: FnMut() {
    f();
    //~^ ERROR: cannot borrow
}
