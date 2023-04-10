//@only-on-host

#[repr(C)]
struct Foo {
    a: i32,
    b: f32,
}

extern "C" {
    fn single_deref(p: *const i32) -> i32;
    fn double_deref(p: *const *const i32) -> i32;
    fn struct_a(p: *const Foo) -> i32;
    fn struct_b(p: *const Foo) -> f32;
}

fn main() {
    let foo = Foo { a: -10, b: -1.0 };

    let a = &20 as *const _;
    let b = &a as *const _;

    let alloc = Box::new(20);
    let alloc_ptr = &(&*alloc as *const _) as *const _;

    unsafe {
        assert_eq!(single_deref(&10), 10);

        println!("Double Alloc");
        assert_eq!(double_deref(alloc_ptr), 20);

        println!("Double Global");
        assert_eq!(double_deref(b), 20);

        assert_eq!(struct_a(&foo), -10);

        assert_eq!(struct_b(&foo), -1.0);
    }
}
