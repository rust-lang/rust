// Check that you can cast between different pointers to trait objects
// whose vtable have the same kind (both lengths, or both trait pointers).

#![feature(unsized_tuple_coercion)]

trait Foo<T> {
    fn foo(&self, _: T) -> u32 { 42 }
}

trait Bar {
    fn bar(&self) { println!("Bar!"); }
}

impl<T> Foo<T> for () {}
impl Foo<u32> for u32 { fn foo(&self, _: u32) -> u32 { self+43 } }
impl Bar for () {}

unsafe fn round_trip_and_call<'a>(t: *const (dyn Foo<u32>+'a)) -> u32 {
    let foo_e : *const dyn Foo<u16> = t as *const _;
    let r_1 = foo_e as *mut dyn Foo<u32>;

    (&*r_1).foo(0)
}

#[repr(C)]
struct FooS<T:?Sized>(T);
#[repr(C)]
struct BarS<T:?Sized>(T);

fn foo_to_bar<T:?Sized>(u: *const FooS<T>) -> *const BarS<T> {
    u as *const BarS<T>
}

fn tuple_i32_to_u32<T:?Sized>(u: *const (i32, T)) -> *const (u32, T) {
    u as *const (u32, T)
}


fn main() {
    let x = 4u32;
    let y : &dyn Foo<u32> = &x;
    let fl = unsafe { round_trip_and_call(y as *const dyn Foo<u32>) };
    assert_eq!(fl, (43+4));

    let s = FooS([0,1,2]);
    let u: &FooS<[u32]> = &s;
    let u: *const FooS<[u32]> = u;
    let bar_ref : *const BarS<[u32]> = foo_to_bar(u);
    let z : &BarS<[u32]> = unsafe{&*bar_ref};
    assert_eq!(&z.0, &[0,1,2]);

    // this assumes that tuple reprs for (i32, _) and (u32, _) are
    // the same.
    let s = (0i32, [0, 1, 2]);
    let u: &(i32, [u8]) = &s;
    let u: *const (i32, [u8]) = u;
    let u_u32 : *const (u32, [u8]) = tuple_i32_to_u32(u);
    unsafe {
        assert_eq!(&(*u_u32).1, &[0, 1, 2]);
    }
}
