enum Foo {
    Bar(u8)
}
fn main(){
    foo(|| {
        match Foo::Bar(1) {
            Foo::Baz(..) => (),
            //~^ ERROR no variant or associated item named `Baz` found
            _ => (),
        }
    });
}

fn foo<F>(f: F) where F: FnMut() {
    f();
    //~^ ERROR: cannot borrow
}
