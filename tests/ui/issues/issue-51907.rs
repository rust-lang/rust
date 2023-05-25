// run-pass
trait Foo {
    extern "C" fn borrow(&self);
    extern "C" fn take(self: Box<Self>);
}

#[repr(C)]
struct Bar {
    val: u8,
}

impl Foo for Bar {
    extern "C" fn borrow(&self) {}
    extern "C" fn take(self: Box<Self>) {}
}

fn main() {
    let foo: Box<dyn Foo> = Box::new(Bar { val: 0 });
    foo.borrow();
    foo.take()
}
