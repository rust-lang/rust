// no-prefer-dynamic

static mut DROP_RAN: bool = false;

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { DROP_RAN = true; }
    }
}

trait Trait { fn dummy(&self) { } }
impl Trait for Foo {}

struct Fat<T: ?Sized> {
    f: T
}

pub fn main() {
    {
        let _x: Box<Fat<Trait>> = Box::<Fat<Foo>>::new(Fat { f: Foo });
    }
    unsafe {
        assert!(DROP_RAN);
    }
}
