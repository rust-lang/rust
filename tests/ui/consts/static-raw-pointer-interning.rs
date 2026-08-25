//@ run-pass

static FOO: Foo = Foo {
    field: &42 as *const i32,
};

struct Foo {
    field: *const i32,
}

unsafe impl Sync for Foo {}

fn main() {
    assert_eq!(unsafe { *FOO.field }, 42);
}
