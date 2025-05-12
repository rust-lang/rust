//@ run-pass

#![feature(ptr_metadata)]

trait Foo {
    fn foo(&self) {} //~ WARN method `foo` is never used
}

struct Bar;

impl Foo for Bar {}

fn main() {
    // Test we can turn a fat pointer to array back into a thin pointer.
    let a: *const [i32] = &[1, 2, 3];
    let b = a as *const [i32; 2];
    unsafe {
        assert_eq!(*b, [1, 2]);
    }

    // Test conversion to an address (usize).
    let a: *const [i32; 3] = &[1, 2, 3];
    let b: *const [i32] = a;
    assert_eq!(a as usize, b as *const () as usize);

    // And conversion to a void pointer/address for trait objects too.
    let a: *mut dyn Foo = &mut Bar;
    let b = a as *mut () as usize;
    let c = a as *const () as usize;
    let d = a.to_raw_parts().0 as usize;

    assert_eq!(b, d);
    assert_eq!(c, d);
}
