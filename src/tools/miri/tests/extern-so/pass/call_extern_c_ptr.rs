//@only-target-linux
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

    unsafe {
        assert_eq!(single_deref(&10), 10);

        assert_eq!(double_deref(&&20), 20);

        assert_eq!(struct_a(&foo), -10);

        assert_eq!(struct_b(&foo), -1.0);
    }
}
