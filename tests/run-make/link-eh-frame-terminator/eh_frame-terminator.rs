//@ run-pass

#[derive(Clone, Copy)]
struct Foo {
    array: [u64; 10240],
}

impl Foo {
    const fn new() -> Self {
        Self { array: [0x1122_3344_5566_7788; 10240] }
    }
}

static BAR: [Foo; 10240] = [Foo::new(); 10240];

fn main() {
    let bt = std::backtrace::Backtrace::force_capture();
    println!("Hello, world! {:?}", bt);
    println!("{:x}", BAR[0].array[0]);
}
