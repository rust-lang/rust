// This test should pass since 'identity' is const fn.

// compile-pass

#![feature(convert_id)]

fn main() {
    const _FOO: u8 = ::std::convert::identity(42u8);
}
