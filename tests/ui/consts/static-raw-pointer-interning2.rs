// run-pass

static mut FOO: Foo = Foo {
    field: &mut [42] as *mut [i32] as *mut i32,
};

struct Foo {
    field: *mut i32,
}

unsafe impl Sync for Foo {}

fn main() {
    assert_eq!(unsafe { *FOO.field = 69; *FOO.field }, 69);
}
