#![allow(dead_code)]

static mut FOO: i32 = 42;
static BAR: Foo = Foo(unsafe { &FOO as *const _} );

struct Foo(*const i32);

unsafe impl Sync for Foo {}

fn main() {
    unsafe {
        assert_eq!(*BAR.0, 42);
        FOO = 5;
        assert_eq!(FOO, 5);
        assert_eq!(*BAR.0, 5);
    }
}
