// This test should pass since we've opted into 'identity' as an
// unstable const fn.

// compile-pass

#![feature(convert_id, const_convert_id)]

fn main() {
    const _FOO: u8 = ::std::convert::identity(42u8);
}
