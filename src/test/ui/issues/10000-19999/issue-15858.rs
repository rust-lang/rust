// run-pass
static mut DROP_RAN: bool = false;

trait Bar {
    fn do_something(&mut self);
}

struct BarImpl;

impl Bar for BarImpl {
    fn do_something(&mut self) {}
}


struct Foo<B: Bar>(B);

impl<B: Bar> Drop for Foo<B> {
    fn drop(&mut self) {
        unsafe {
            DROP_RAN = true;
        }
    }
}


fn main() {
    {
       let _x: Foo<BarImpl> = Foo(BarImpl);
    }
    unsafe {
        assert_eq!(DROP_RAN, true);
    }
}
