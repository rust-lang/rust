//@ known-bug: rust-lang/rust #124464
enum TestOption<T> {
    TestSome(T),
    TestSome(T),
}

pub struct Request {
    bar: TestOption<u64>,
    bar: u8,
}

fn default_instance() -> &'static Request {
    static instance: Request = Request { bar: 17 };
    &instance
}

pub fn main() {}
