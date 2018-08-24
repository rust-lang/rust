// no-prefer-dynamic
// ignore-emscripten

thread_local!(static FOO: Foo = Foo);
thread_local!(static BAR: Bar = Bar(1));
thread_local!(static BAZ: Baz = Baz);

static mut HIT: bool = false;

struct Foo;
struct Bar(i32);
struct Baz;

impl Drop for Foo {
    fn drop(&mut self) {
        BAR.with(|_| {});
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        assert_eq!(self.0, 1);
        self.0 = 2;
        BAZ.with(|_| {});
        assert_eq!(self.0, 2);
    }
}

impl Drop for Baz {
    fn drop(&mut self) {
        unsafe { HIT = true; }
    }
}

fn main() {
    std::thread::spawn(|| {
        FOO.with(|_| {});
    }).join().unwrap();
    assert!(unsafe { HIT });
}
