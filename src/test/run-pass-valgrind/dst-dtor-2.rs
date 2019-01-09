// no-prefer-dynamic

static mut DROP_RAN: isize = 0;

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { DROP_RAN += 1; }
    }
}

struct Fat<T: ?Sized> {
    f: T
}

pub fn main() {
    {
        let _x: Box<Fat<[Foo]>> = Box::<Fat<[Foo; 3]>>::new(Fat { f: [Foo, Foo, Foo] });
    }
    unsafe {
        assert_eq!(DROP_RAN, 3);
    }
}
