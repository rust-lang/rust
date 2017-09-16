// Moving around undef is not allowed by validation
// compile-flags: -Zmir-emit-validate=0

struct Foo {
    _inner: i32,
}

fn main() {
    unsafe {
        let foo = Foo {
            _inner: std::mem::uninitialized(),
        };
        let _bar = foo;
    }
}
