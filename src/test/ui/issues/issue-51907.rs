// run-pass
trait Foo {
    extern fn borrow(&self);
    extern fn take(self: Box<Self>);
}

struct Bar;
impl Foo for Bar {
    #[allow(improper_ctypes_definitions)]
    extern fn borrow(&self) {}
    #[allow(improper_ctypes_definitions)]
    extern fn take(self: Box<Self>) {}
}

fn main() {
    let foo: Box<dyn Foo> = Box::new(Bar);
    foo.borrow();
    foo.take()
}
