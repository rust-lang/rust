// no-prefer-dynamic

#![feature(unsized_tuple_coercion)]

static mut DROP_RAN: bool = false;

struct Foo;
impl Drop for Foo {
    fn drop(&mut self) {
        unsafe { DROP_RAN = true; }
    }
}

trait Trait { fn dummy(&self) { } }
impl Trait for Foo {}

pub fn main() {
    {
        let _x: Box<(i32, Trait)> = Box::<(i32, Foo)>::new((42, Foo));
    }
    unsafe {
        assert!(DROP_RAN);
    }
}
