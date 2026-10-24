#![feature(extern_types)]

extern "C" {
    type Foo;
    static FOO: Foo;
    fn bar(foo: *const Foo) -> u8;
}

fn main() {
    unsafe {
        let foo = &FOO;
        assert_eq!(bar(foo), 42);
    }
}
