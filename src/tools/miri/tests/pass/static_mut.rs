static mut FOO: i32 = 42;

// FIXME: Use `SyncUnsafeCell` instead of allowing `static_mut_refs` lint
#[allow(static_mut_refs)]
static BAR: Foo = Foo(unsafe { &FOO as *const _ });

#[allow(dead_code)]
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
