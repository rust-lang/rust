//@ run-pass
// Check that you can cast between different pointers to trait objects
// whose vtable have the same kind (both lengths, or both trait pointers).

trait Bar { //~ WARN trait `Bar` is never used
    fn bar(&self) { println!("Bar!"); }
}

impl Bar for () {}

#[repr(C)]
struct FooS<T:?Sized>(T);
#[repr(C)]
struct BarS<T:?Sized>(T);

fn foo_to_bar<T:?Sized>(u: *const FooS<T>) -> *const BarS<T> {
    u as *const BarS<T>
}


fn main() {
    let s = FooS([0,1,2]);
    let u: &FooS<[u32]> = &s;
    let u: *const FooS<[u32]> = u;
    let bar_ref : *const BarS<[u32]> = foo_to_bar(u);
    let z : &BarS<[u32]> = unsafe{&*bar_ref};
    assert_eq!(&z.0, &[0,1,2]);
}
