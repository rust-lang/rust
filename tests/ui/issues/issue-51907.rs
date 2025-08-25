//@ run-pass
trait Foo {
    #[allow(improper_c_fn_definitions)]
    extern "C" fn borrow(&self);
    #[allow(improper_c_fn_definitions)]
    extern "C" fn take(self: Box<Self>);
}

struct Bar;
impl Foo for Bar {
    #[allow(improper_c_fn_definitions)]
    extern "C" fn borrow(&self) {}
    #[allow(improper_c_fn_definitions)]
    extern "C" fn take(self: Box<Self>) {}
}

fn main() {
    let foo: Box<dyn Foo> = Box::new(Bar);
    foo.borrow();
    foo.take()
}
