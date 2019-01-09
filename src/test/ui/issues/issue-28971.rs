// This should not cause an ICE

enum Foo {
    Bar(u8)
}
fn main(){
    foo(|| {
        match Foo::Bar(1) {
            Foo::Baz(..) => (),
            //~^ ERROR no variant named `Baz` found for type `Foo`
            _ => (),
        }
    });
}

fn foo<F>(f: F) where F: FnMut() {
    f();
}
