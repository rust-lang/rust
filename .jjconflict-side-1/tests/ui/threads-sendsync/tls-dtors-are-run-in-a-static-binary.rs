//@ run-pass
//@ no-prefer-dynamic
//@ needs-threads

static mut HIT: bool = false;

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        unsafe {
            HIT = true;
        }
    }
}

thread_local!(static FOO: Foo = Foo);

fn main() {
    std::thread::spawn(|| {
        FOO.with(|_| {});
    })
    .join()
    .unwrap();
    assert!(unsafe { HIT });
}
