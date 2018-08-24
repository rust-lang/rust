// This test should fail since identity is not stable as a const fn yet.

#![feature(convert_id)]

fn main() {
    const _FOO: u8 = ::std::convert::identity(42u8);
}
