// Don't const eval fields with ambiguous layout.
// See issues #125842 and #124464.

enum TestOption<T> {
    TestSome(T),
    TestSome(T),
//~^ ERROR the name `TestSome` is defined multiple times
}

pub struct Request {
    bar: TestOption<u64>,
    bar: u8,
//~^ ERROR field `bar` is already declared
}

fn default_instance() -> &'static Request {
    static instance: Request = Request { bar: 17 };
    &instance
}

pub fn main() {}
