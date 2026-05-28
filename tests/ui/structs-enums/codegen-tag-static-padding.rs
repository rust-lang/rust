//@ run-pass
#![allow(non_upper_case_globals)]

// Issue #13186

// For simplicity of explanations assuming code is compiled for x86_64
// Linux ABI.

// Size of TestOption<u64> is 16, and alignment of TestOption<u64> is 8.
// Size of u8 is 1, and alignment of u8 is 1.
// So size of Request is 24, and alignment of Request must be 8:
// the maximum alignment of its fields.
// Last 7 bytes of Request struct are not occupied by any fields.



enum TestOption<T> {
    TestNone,
    TestSome(T),
}

pub struct Request {
    foo: TestOption<u64>,
    bar: u8,
}

fn default_instance() -> &'static Request {
    static instance: Request = Request {
        // LLVM does not allow to specify alignment of expressions, thus
        // alignment of `foo` in constant is 1, not 8.
        foo: TestOption::TestNone,
        bar: 17,
        // Space after last field is not occupied by any data, but it is
        // reserved to make struct aligned properly. If compiler does
        // not insert padding after last field when emitting constant,
        // size of struct may be not equal to size of struct, and
        // compiler crashes in internal assertion check.
    };
    &instance
}

fn non_default_instance() -> &'static Request {
    static instance: Request = Request {
        foo: TestOption::TestSome(0x1020304050607080),
        bar: 19,
    };
    &instance
}

pub fn main() {
    match default_instance() {
        &Request { foo: TestOption::TestNone, bar: 17 } => {},
        _ => panic!(),
    };
    match non_default_instance() {
        &Request { foo: TestOption::TestSome(0x1020304050607080), bar: 19 } => {},
        _ => panic!(),
    };
}
